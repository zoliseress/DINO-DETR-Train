import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from datamodule import load_dino
from matcher import hungarian_matching
from detr import DETR


class DETR_Lightning(pl.LightningModule):
    def __init__(self, config):
        """
        Initialize DETR Lightning Module.
        
        Args:
            config: Configuration dictionary (contains freeze_backbone and unfreeze_backbone_at_epoch)
        """

        super().__init__()

        # Save hyperparameters to hparams.yaml
        self.save_hyperparameters(ignore=['config'])

        nc = config["model"]["num_classes"]  # COCO example
        backbone, backbone_channels = load_dino(config["model"]["backbone"])

        self.model = DETR(
            backbone,
            backbone_channels,
            num_classes=nc,
            hidden_dim=config["model"]["hidden_dim"],
            num_queries=config["model"]["num_queries"],
        )

        # Get fine-tuning parameters from config
        freeze_backbone = config["train"].get("freeze_backbone", True)
        self.unfreeze_backbone_at_epoch = config["train"].get("unfreeze_backbone_at_epoch", -1)
        
        # Track backbone freeze state
        self.backbone_frozen = freeze_backbone
        
        if freeze_backbone:
            self.freeze_backbone()

        self.learning_rate = config["train"]["learning_rate"]
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
        if (self.unfreeze_backbone_at_epoch >= 0 and 
            self.current_epoch >= self.unfreeze_backbone_at_epoch and 
            self.backbone_frozen):
            self.unfreeze_backbone()

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

    def training_step(self, batch, batch_idx):

        images, targets = batch
        preds = self(images)
        
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

        # IMPORTANT: pass model predictions to the matcher (not the GT tensors)
        matched_indices = hungarian_matching(
            preds["pred_logits"], preds["pred_boxes"], batch_targets, device=self.device
        )

        batch_size = len(batch_targets)
        num_queries = preds["pred_logits"].shape[1]

        # Accumulate losses
        cls_losses = []
        l1_losses = []
        giou_losses = []
        unmatched_cls_losses = []

        # Track which queries are matched in each image
        matched_query_ids = [set() for _ in range(batch_size)]

        # Process matched pairs.
        for b_idx, pairs in enumerate(matched_indices):
            for pred_idx, gt_idx in pairs:

                matched_query_ids[b_idx].add(pred_idx)

                # Classification loss.
                pred_logit = preds["pred_logits"][b_idx, pred_idx]  # [num_classes]
                gt_class = int(batch_targets[b_idx][gt_idx]["category_id"])
                loss_cls = self.criterion(pred_logit.unsqueeze(0), torch.tensor([gt_class], device=self.device, dtype=torch.long))
                cls_losses.append(loss_cls)

                # BBox losses: L1 + GIoU
                # The pred boxes are in COCO format: [x, y, w, h] normalized to [0, 1].
                pred_box_norm = preds["pred_boxes"][b_idx, pred_idx]  # [4] [x, y, w, h] normalized
                
                # GT bbox is already in center format [cx, cy, w, h] in pixel coords
                # Convert to normalized [0, 1]
                gt_bbox_pixel = batch_targets[b_idx][gt_idx]["bbox"]  # [cx, cy, w, h] in 518x518 space
                gt_bbox_norm = torch.tensor(
                    gt_bbox_pixel, device=self.device, dtype=torch.float32
                ) / 518.0  # [cx, cy, w, h] normalized to [0, 1]

                # L1 loss: directly on [cx, cy, w, h] format
                loss_l1 = self.l1_loss(pred_box_norm, gt_bbox_norm)
                l1_losses.append(loss_l1)

                # GIoU loss: convert both to xyxy format
                # pred_box_norm is [cx, cy, w, h]
                pred_cx, pred_cy, pred_w, pred_h = pred_box_norm
                pred_xyxy = torch.stack([
                    pred_cx - pred_w / 2,
                    pred_cy - pred_h / 2,
                    pred_cx + pred_w / 2,
                    pred_cy + pred_h / 2
                ]).clamp(0, 1).unsqueeze(0)
                
                # gt_bbox_norm is [cx, cy, w, h]
                gt_cx, gt_cy, gt_w, gt_h = gt_bbox_norm
                gt_xyxy = torch.stack([
                    gt_cx - gt_w / 2,
                    gt_cy - gt_h / 2,
                    gt_cx + gt_w / 2,
                    gt_cy + gt_h / 2
                ]).clamp(0, 1).unsqueeze(0)
                
                loss_giou = self.generalized_box_iou(pred_xyxy, gt_xyxy)
                giou_losses.append(loss_giou)
                
                #####
                # from torchvision.ops import generalized_box_iou
                # loss_giou = 1 - generalized_box_iou(boxes1_xyxy, boxes2_xyxy).mean()
                #####

        # Process unmatched queries: force them to predict background class
        # Background class is index num_classes (COCO uses 91 classes, so background is 91)
        num_classes = preds["pred_logits"].shape[2]
        background_class = num_classes - 1  # Last class is background
        
        for b_idx in range(batch_size):
            for query_idx in range(num_queries):
                if query_idx not in matched_query_ids[b_idx]:
                    # Unmatched query: should predict background
                    pred_logit = preds["pred_logits"][b_idx, query_idx]  # [num_classes]
                    loss_cls_bg = self.criterion(pred_logit.unsqueeze(0), torch.tensor([background_class], device=self.device, dtype=torch.long))
                    unmatched_cls_losses.append(loss_cls_bg)

        # Average losses
        if cls_losses:
            loss_cls = torch.stack(cls_losses).mean()
        else:
            loss_cls = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
        if l1_losses:
            loss_l1 = torch.stack(l1_losses).mean()
        else:
            loss_l1 = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
        if giou_losses:
            loss_giou = torch.stack(giou_losses).mean()
        else:
            loss_giou = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if unmatched_cls_losses:
            loss_unmatched_cls = torch.stack(unmatched_cls_losses).mean()
        else:
            loss_unmatched_cls = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Combine losses with weights.
        loss_bbox = self.loss_l1_weight * loss_l1 + self.loss_giou_weight * loss_giou
        loss = (loss_cls + self.loss_no_object * loss_unmatched_cls) + self.loss_bbox_weight * loss_bbox

        # Clamp and check for NaN.
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf detected in loss at step {self.global_step}")
            print(f"  loss_cls={loss_cls}, loss_unmatched_cls={loss_unmatched_cls}")
            print(f"  loss_l1={loss_l1}, loss_giou={loss_giou}, loss_bbox={loss_bbox}")
            # Return zero loss to skip this batch (prevents training crash)
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)

        # Logging.
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        lr = self.lr_schedulers().optimizer.param_groups[0]["lr"]
        self.log('learning_rate', lr, on_step=False, on_epoch=True)
        
        if isinstance(loss_cls, torch.Tensor):
            self.log('train_loss_cls', loss_cls, on_step=False, on_epoch=True)
        if isinstance(loss_unmatched_cls, torch.Tensor):
            self.log('train_loss_unmatched_cls', loss_unmatched_cls, on_step=False, on_epoch=True)
        if isinstance(loss_l1, torch.Tensor):
            self.log('train_loss_l1', loss_l1, on_step=False, on_epoch=True)
        if isinstance(loss_giou, torch.Tensor):
            self.log('train_loss_giou', loss_giou, on_step=False, on_epoch=True)
        if isinstance(loss_bbox, torch.Tensor):
            self.log('train_loss_bbox', loss_bbox, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        images, targets = batch
        preds = self(images)
        
        loss_class = self.criterion(preds["pred_logits"].view(-1, preds["pred_logits"].shape[-1]), targets["labels"])
        loss_bbox = self.bbox_loss_fn(preds["pred_boxes"].view(-1, 4), targets["boxes"])
        loss = loss_class + loss_bbox
        
        self.log('val_loss', loss)
        self.val_accuracy(preds["pred_logits"].argmax(-1), targets["labels"])
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        """
        # Different LR can be set to backbone and head (if the backbone is not freezed).
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            min_lr=1e-7,
        )

        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'monitor': 'train_loss',
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
