import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

from datamodule import load_backbone
from matcher import hungarian_matching
from detr import DETR
from training_diagnostics import TrainingDiagnostics


class DETR_Lightning(TrainingDiagnostics, pl.LightningModule):
    def __init__(self, config: dict):
        """
        Initialize DETR Lightning Module.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """

        super().__init__()

        # Save hyperparameters to hparams.yaml
        self.save_hyperparameters()

        nc = config["model"]["num_classes"]
        backbone, backbone_channels = load_backbone(config["model"]["backbone"])

        self.model = DETR(
            backbone,
            backbone_channels,
            num_classes=nc,
            hidden_dim=config["model"]["hidden_dim"],
            num_queries=config["model"]["num_queries"],
        )

        # Get fine-tuning parameters from config
        freeze_backbone = config["train"].get("freeze_backbone", True)
        self.unfreeze_backbone_at_epoch = config["train"].get(
            "unfreeze_backbone_at_epoch", -1
        )
        
        # Track backbone freeze state
        self.backbone_frozen = freeze_backbone
        
        if freeze_backbone:
            self.freeze_backbone()

        self.learning_rate = config["train"]["learning_rate"]
        self.backbone_learning_rate = config["train"].get(
            "backbone_learning_rate", self.learning_rate * 0.1
        )
        self.weight_decay = config["train"].get("weight_decay", 1e-4)
        self.debug_log_on_step = config["train"].get("debug_log_on_step", True)
        self.debug_log_every_n_steps = int(config["train"].get("debug_log_every_n_steps", 25))
        if self.debug_log_every_n_steps <= 0:
            self.debug_log_every_n_steps = 1

        # Scheduler defaults are validation-driven to avoid overfitting to train loss.
        train_cfg = config.get("train", {})
        val_monitor_enabled = bool(config.get("validation_monitor", {}).get("enabled", True))
        default_monitor = "val_epoch_score" if val_monitor_enabled else "train_loss"
        self.lr_scheduler_monitor = str(train_cfg.get("lr_scheduler_monitor", default_monitor))
        if self.lr_scheduler_monitor.endswith("loss") or self.lr_scheduler_monitor == "val_epoch_score":
            default_mode = "min"
        elif self.lr_scheduler_monitor.startswith("val_"):
            default_mode = "max"
        else:
            default_mode = "min"
        self.lr_scheduler_mode = str(train_cfg.get("lr_scheduler_mode", default_mode))
        self.lr_scheduler_factor = float(train_cfg.get("lr_scheduler_factor", 0.1))
        self.lr_scheduler_patience = int(train_cfg.get("lr_scheduler_patience", 10))
        self.lr_scheduler_min_lr = float(train_cfg.get("lr_scheduler_min_lr", 1e-7))

        # Lightweight training diagnostics.
        self.profile_diagnostics_enabled = bool(train_cfg.get("profile_diagnostics_enabled", True))
        self.profile_memory_enabled = self.profile_diagnostics_enabled and bool(
            train_cfg.get("profile_memory_enabled", True)
        )
        self.profile_timing_enabled = self.profile_diagnostics_enabled and bool(
            train_cfg.get("profile_timing_enabled", True)
        )
        self.profile_timing_cuda_sync = bool(train_cfg.get("profile_timing_cuda_sync", False))
        self.profile_timing_print = bool(train_cfg.get("profile_timing_print", True))
        self._reset_timing_counters()

        self.criterion = torch.nn.CrossEntropyLoss()

        # L1 loss for bounding box regression
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        
        # Loss weights (DETR paper uses these ratios)
        self.loss_bbox_weight = config["model"]["weight_bbox"]
        self.loss_l1_weight = config["model"]["weight_L1"]
        self.loss_giou_weight = config["model"]["weight_giou"]
        self.loss_no_object = config["model"]["weight_no_object"]
        
        self.train_accuracy = Accuracy(task='multiclass', num_classes=nc)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=nc)
        self._printed_psutil_warning = False

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for p in self.model.backbone.parameters():
            p.requires_grad = True
        self.backbone_frozen = False
        print("Backbone unfrozen - fine-tuning backbone")

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        """Called at the start of each training epoch. Used to unfreeze backbone at specified epoch."""
        self._reset_timing_counters()
        if (self.unfreeze_backbone_at_epoch >= 0 and 
            self.current_epoch >= self.unfreeze_backbone_at_epoch and 
            self.backbone_frozen):
            self.unfreeze_backbone()

    def on_train_epoch_end(self):
        """Log system RAM and GPU VRAM usage after each epoch."""
        self._log_memory_usage()
        self._log_timing_usage()

    def _should_log_debug_step(self) -> bool:
        """Return True when a debug step metric should be emitted."""
        return self.debug_log_on_step and (self.global_step % self.debug_log_every_n_steps == 0)

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU loss.
        boxes1, boxes2: [N, 4] in (x0, y0, x1, y1) normalized format
        Returns: GIoU loss (1 - GIoU)
        """
        # Swap coordinates if needed to ensure x0 <= x1, y0 <= y1
        boxes1 = torch.stack([
            torch.min(boxes1[:, 0], boxes1[:, 2]),
            torch.min(boxes1[:, 1], boxes1[:, 3]),
            torch.max(boxes1[:, 0], boxes1[:, 2]),
            torch.max(boxes1[:, 1], boxes1[:, 3])
        ], dim=1)
        
        boxes2 = torch.stack([
            torch.min(boxes2[:, 0], boxes2[:, 2]),
            torch.min(boxes2[:, 1], boxes2[:, 3]),
            torch.max(boxes2[:, 0], boxes2[:, 2]),
            torch.max(boxes2[:, 1], boxes2[:, 3])
        ], dim=1)
        
        # Compute intersection with numerical stability
        inter_x0 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y0 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x1 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y1 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_w = (inter_x1 - inter_x0).clamp(min=0)
        inter_h = (inter_y1 - inter_y0).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1 + area2 - inter_area
        
        # IoU with epsilon to avoid division by zero
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        # Enclosing box
        enclose_x0 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enclose_y0 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enclose_x1 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enclose_y1 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enclose_area = (enclose_x1 - enclose_x0) * (enclose_y1 - enclose_y0)
        
        # GIoU with clamped division
        giou_gap = (enclose_area - union_area) / torch.clamp(enclose_area, min=1e-6)
        giou = iou - giou_gap
        
        # Clamp to valid range
        giou = torch.clamp(giou, min=-1.0, max=1.0)
        
        return torch.mean(1.0 - giou)

    def _compute_single_output_losses(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        batch_targets,
        images: torch.Tensor,
    ):
        """Compute DETR losses for one decoder output tensor pair."""

        matched_indices = hungarian_matching(
            pred_logits,
            pred_boxes,
            batch_targets,
            device=self.device,
        )

        batch_size = len(batch_targets)
        num_queries = pred_logits.shape[1]

        matched_batch_indices = []
        matched_pred_indices = []
        matched_gt_boxes_pixel = []
        matched_gt_classes = []

        for b_idx, pairs in enumerate(matched_indices):
            for pred_idx, gt_idx in pairs:
                matched_batch_indices.append(int(b_idx))
                matched_pred_indices.append(int(pred_idx))
                ann = batch_targets[b_idx][gt_idx]
                matched_gt_boxes_pixel.append(ann["bbox"])
                matched_gt_classes.append(int(ann["category_id"]))

        num_classes = pred_logits.shape[2]
        background_class = num_classes - 1
        target_classes = torch.full(
            (batch_size, num_queries),
            fill_value=background_class,
            dtype=torch.long,
            device=self.device,
        )

        if matched_batch_indices:
            matched_batch_idx_tensor = torch.as_tensor(
                matched_batch_indices,
                dtype=torch.long,
                device=self.device,
            )
            matched_pred_idx_tensor = torch.as_tensor(
                matched_pred_indices,
                dtype=torch.long,
                device=self.device,
            )
            matched_gt_classes_tensor = torch.as_tensor(
                matched_gt_classes,
                dtype=torch.long,
                device=self.device,
            )
            target_classes[matched_batch_idx_tensor, matched_pred_idx_tensor] = matched_gt_classes_tensor

        class_weights = torch.ones(num_classes, device=self.device, dtype=torch.float32)
        class_weights[background_class] = self.loss_no_object

        loss_cls = F.cross_entropy(
            pred_logits.permute(0, 2, 1),
            target_classes,
            weight=class_weights,
        )

        if matched_batch_indices:
            src_boxes = pred_boxes[matched_batch_idx_tensor, matched_pred_idx_tensor]

            img_h, img_w = images.shape[-2], images.shape[-1]
            scale = src_boxes.new_tensor([img_w, img_h, img_w, img_h])
            target_boxes_for_loss = torch.as_tensor(
                matched_gt_boxes_pixel,
                device=self.device,
                dtype=src_boxes.dtype,
            ) / scale

            num_boxes = src_boxes.shape[0]
            loss_l1 = F.l1_loss(
                src_boxes,
                target_boxes_for_loss,
                reduction="none",
            ).sum() / num_boxes

            src_xyxy = torch.cat(
                (
                    src_boxes[:, :2] - src_boxes[:, 2:] / 2,
                    src_boxes[:, :2] + src_boxes[:, 2:] / 2,
                ),
                dim=1,
            ).clamp(0, 1)
            tgt_xyxy = torch.cat(
                (
                    target_boxes_for_loss[:, :2] - target_boxes_for_loss[:, 2:] / 2,
                    target_boxes_for_loss[:, :2] + target_boxes_for_loss[:, 2:] / 2,
                ),
                dim=1,
            ).clamp(0, 1)
            loss_giou = self.generalized_box_iou(src_xyxy, tgt_xyxy)
        else:
            loss_l1 = pred_logits.new_tensor(0.0)
            loss_giou = pred_logits.new_tensor(0.0)

        loss_bbox = self.loss_l1_weight * loss_l1 + self.loss_giou_weight * loss_giou
        loss = loss_cls + self.loss_bbox_weight * loss_bbox

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_l1": loss_l1,
            "loss_giou": loss_giou,
            "loss_bbox": loss_bbox,
            "matched_indices": matched_indices,
            "target_classes": target_classes,
            "background_class": background_class,
        }

    def training_step(self, batch, batch_idx):

        step_start_ts = 0.0
        if self.profile_timing_enabled:
            step_start_ts = self._mark_time()

        images, targets = batch

        if self.profile_timing_enabled:
            t0 = self._mark_time()

        preds = self(images)  # Forward pass.

        if self.profile_timing_enabled:
            self._timing_forward_sum += self._elapsed_since(t0)
        
        # Normalize targets into a batch-list: dataloader sometimes returns
        # a list-of-annotations (for batch_size>1) or a single list (for batch_size==1).
        if isinstance(targets, list) and len(targets) and isinstance(targets[0], dict):
            # single image: targets is a list of annotation dicts -> wrap into batch
            batch_targets = [targets]
        elif isinstance(targets, list) and len(targets) and isinstance(targets[0], list):
            # already batched: list of lists
            batch_targets = targets
        else:
            batch_targets = [targets]

        # Main decoder output loss.
        if self.profile_timing_enabled:
            t0 = self._mark_time()
        
        main_losses = self._compute_single_output_losses(
            pred_logits=preds["pred_logits"],
            pred_boxes=preds["pred_boxes"],
            batch_targets=batch_targets,
            images=images,
        )

        if self.profile_timing_enabled:
            self._timing_main_loss_sum += self._elapsed_since(t0)

        loss = main_losses["loss"]
        loss_cls = main_losses["loss_cls"]
        loss_l1 = main_losses["loss_l1"]
        loss_giou = main_losses["loss_giou"]
        loss_bbox = main_losses["loss_bbox"]
        matched_indices = main_losses["matched_indices"]
        target_classes = main_losses["target_classes"]
        background_class = main_losses["background_class"]

        # Auxiliary decoder supervision: apply the same loss to intermediate decoder outputs.
        aux_outputs = preds.get("aux_outputs", [])
        if self.profile_timing_enabled:
            t0 = self._mark_time()
        if aux_outputs:
            aux_loss_values = []
            for aux_pred in aux_outputs:
                aux_losses = self._compute_single_output_losses(
                    pred_logits=aux_pred["pred_logits"],
                    pred_boxes=aux_pred["pred_boxes"],
                    batch_targets=batch_targets,
                    images=images,
                )
                aux_loss_values.append(aux_losses["loss"])

            aux_loss = torch.stack(aux_loss_values).sum()
            # Official DETR-style aggregation sums final and auxiliary losses.
            loss = main_losses["loss"] + aux_loss
        else:
            aux_loss = preds["pred_logits"].new_tensor(0.0)

        if self.profile_timing_enabled:
            self._timing_aux_loss_sum += self._elapsed_since(t0)

        # Clamp and check for NaN.
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf detected in loss at step {self.global_step}")
            print(f"  loss_cls={loss_cls}")
            print(f"  loss_l1={loss_l1}, loss_giou={loss_giou}, loss_bbox={loss_bbox}, aux_loss={aux_loss}")
            # Return zero loss to skip this batch (prevents training crash)
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)

        log_debug_step = self._should_log_debug_step()

        # Logging.
        if self.profile_timing_enabled:
            t0 = self._mark_time()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if log_debug_step:
            self.log('step_train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        if self.trainer is not None and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            head_lr = optimizer.param_groups[0]["lr"]
            self.log('learning_rate', head_lr, on_step=False, on_epoch=True)
            self.log('learning_rate_head', head_lr, on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_learning_rate', head_lr, on_step=True, on_epoch=False)
                self.log('step_learning_rate_head', head_lr, on_step=True, on_epoch=False)
            if len(optimizer.param_groups) > 1:
                backbone_lr = optimizer.param_groups[1]["lr"]
                self.log('learning_rate_backbone', backbone_lr, on_step=False, on_epoch=True)
                if log_debug_step:
                    self.log('step_learning_rate_backbone', backbone_lr, on_step=True, on_epoch=False)
        
        if isinstance(loss_cls, torch.Tensor):
            self.log('train_loss_cls', loss_cls, on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_train_loss_cls', loss_cls, on_step=True, on_epoch=False)
        if isinstance(loss_l1, torch.Tensor):
            self.log('train_loss_l1', loss_l1, on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_train_loss_l1', loss_l1, on_step=True, on_epoch=False)
        if isinstance(loss_giou, torch.Tensor):
            self.log('train_loss_giou', loss_giou, on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_train_loss_giou', loss_giou, on_step=True, on_epoch=False)
        if isinstance(loss_bbox, torch.Tensor):
            self.log('train_loss_bbox', loss_bbox, on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_train_loss_bbox', loss_bbox, on_step=True, on_epoch=False)
        if aux_outputs:
            self.log('train_loss_aux', aux_loss, on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_train_loss_aux', aux_loss, on_step=True, on_epoch=False)

        # Lightweight debug metrics
        matched_counts = torch.tensor([len(pairs) for pairs in matched_indices], device=self.device, dtype=torch.float32)
        if matched_counts.numel() > 0:
            self.log('train_matched_avg', matched_counts.mean(), on_step=False, on_epoch=True)
            if log_debug_step:
                self.log('step_train_matched_avg', matched_counts.mean(), on_step=True, on_epoch=False)
        unique_gt = set()
        for ann_list in batch_targets:
            for ann in ann_list:
                unique_gt.add(int(ann["category_id"]))
        self.log('train_gt_unique_classes', float(len(unique_gt)), on_step=False, on_epoch=True)
        if log_debug_step:
            self.log('step_train_gt_unique_classes', float(len(unique_gt)), on_step=True, on_epoch=False)

        # Background ratio and prediction diversity (debug)
        bg_ratio = (target_classes == background_class).float().mean()
        self.log('train_bg_ratio', bg_ratio, on_step=False, on_epoch=True)
        pred_logits_std = preds["pred_logits"].std(dim=1).mean()
        pred_boxes_std = preds["pred_boxes"].std(dim=1).mean()
        self.log('train_pred_logits_std', pred_logits_std, on_step=False, on_epoch=True)
        self.log('train_pred_boxes_std', pred_boxes_std, on_step=False, on_epoch=True)
        
        if log_debug_step:
            self.log('step_train_bg_ratio', bg_ratio, on_step=True, on_epoch=False)
            self.log('step_train_pred_logits_std', pred_logits_std, on_step=True, on_epoch=False)
            self.log('step_train_pred_boxes_std', pred_boxes_std, on_step=True, on_epoch=False)

        if self.profile_timing_enabled:
            self._timing_logging_sum += self._elapsed_since(t0)
            self._timing_step_total_sum += self._elapsed_since(step_start_ts)
            self._timing_steps += 1

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)

        # Normalize targets into a batch-list: dataloader may return
        # a list-of-annotations (batch_size==1) or list of lists.
        if isinstance(targets, list) and len(targets) and isinstance(targets[0], dict):
            batch_targets = [targets]
        elif isinstance(targets, list) and len(targets) and isinstance(targets[0], list):
            batch_targets = targets
        else:
            batch_targets = [targets]

        main_losses = self._compute_single_output_losses(
            pred_logits=preds["pred_logits"],
            pred_boxes=preds["pred_boxes"],
            batch_targets=batch_targets,
            images=images,
        )

        val_loss = main_losses["loss"]
        val_loss_cls = main_losses["loss_cls"]
        val_loss_l1 = main_losses["loss_l1"]
        val_loss_giou = main_losses["loss_giou"]
        val_loss_bbox = main_losses["loss_bbox"]

        aux_outputs = preds.get("aux_outputs", [])
        if aux_outputs:
            aux_loss_values = []
            for aux_pred in aux_outputs:
                aux_losses = self._compute_single_output_losses(
                    pred_logits=aux_pred["pred_logits"],
                    pred_boxes=aux_pred["pred_boxes"],
                    batch_targets=batch_targets,
                    images=images,
                )
                aux_loss_values.append(aux_losses["loss"])

            val_aux_loss = torch.stack(aux_loss_values).sum()
            val_loss = val_loss + val_aux_loss
            self.log('val_loss_aux', val_aux_loss, on_step=False, on_epoch=True)

        # Keep monitor key name but use positive, decreasing score (loss-like).
        val_epoch_score = val_loss

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_epoch_score', val_epoch_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_cls', val_loss_cls, on_step=False, on_epoch=True)
        self.log('val_loss_l1', val_loss_l1, on_step=False, on_epoch=True)
        self.log('val_loss_giou', val_loss_giou, on_step=False, on_epoch=True)
        self.log('val_loss_bbox', val_loss_bbox, on_step=False, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        """
        """
        backbone_params = list(self.model.backbone.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": head_params,
                    "lr": self.learning_rate,
                },
                {
                    "params": backbone_params,
                    "lr": self.backbone_learning_rate,
                },
            ],
            weight_decay=self.weight_decay,
        )

        print(
            "Optimizer LRs: "
            f"head={self.learning_rate}, backbone={self.backbone_learning_rate}, "
            f"weight_decay={self.weight_decay}"
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.lr_scheduler_mode,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr,
        )

        print(
            "LR scheduler settings: "
            f"monitor={self.lr_scheduler_monitor}, mode={self.lr_scheduler_mode}, "
            f"factor={self.lr_scheduler_factor}, patience={self.lr_scheduler_patience}, "
            f"min_lr={self.lr_scheduler_min_lr}"
        )

        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'monitor': self.lr_scheduler_monitor,
            'interval': 'epoch',
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def _prepare_targets(self, targets):
        """Convert COCO-style targets list to batched tensors."""

        # targets is a list of lists, where each inner list contains annotation dicts
        batch_size = len(targets)
        max_objects = max(len(ann_list) for ann_list in targets) if targets else 0
               
        # classification labels must be integer (long) for CrossEntropyLoss
        labels = torch.full((batch_size, max_objects), fill_value=0, dtype=torch.long, device=self.device)
        boxes = torch.zeros(batch_size, max_objects, 4, dtype=torch.float32, device=self.device)
        
        for i, ann_list in enumerate(targets):
            for j, ann in enumerate(ann_list):
                labels[i, j] = int(ann["category_id"])
                boxes[i, j] = torch.tensor(ann["bbox"], dtype=torch.float32, device=self.device)

        return labels, boxes
