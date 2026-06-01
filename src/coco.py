import torch
import torchvision.transforms as T

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection


def build_transform(config: dict) -> T.Compose:
    """
    Build a torchvision transform pipeline based on the backbone type and image size
    specified in the config.
        1. ResNet-style preprocessing: resize to fixed size (e.g., 480x480),
           ImageNet normalization.
        2. DINOv2-style preprocessing: resize to multiple of patch size (e.g., 518x518),
           no normalization.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data settings.

    Returns
    -------
    T.Compose
        A torchvision transform pipeline.
    
    """

    backbone_name = str(config["model"]["backbone"]).lower()
    data_cfg = config.get("data", {})

    # ResNet-style preprocessing (fixed-size to keep batch stackable).
    if "resnet" in backbone_name:
        input_size = int(data_cfg.get("input_size", 480))
        return T.Compose([
            T.Resize((input_size, input_size)),  #  (480, 480) for ResNet50/101
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # DINOv2-style preprocessing (resize to multiple of patch size, no normalization).
    input_size = int(data_cfg.get("input_size", 518))
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])


def collate_fn_coco(batch: list) -> tuple[torch.Tensor, list]:
    """
     Custom collate function that scales bounding boxes based on the current
     (already resized) image tensor size.

     1. Images may be resized by the dataset transform (e.g., ResNet-compatible sizes).
     2. DETR uses bounding boxes in center (cx, cy, w, h) format, so we convert
         from corner (x, y, w, h) format to center (cx, cy, w, h) format.

    Parameters
    ----------
    batch : list
        A batch of (image, targets) tuples from the COCO dataset.

    Returns
    -------
    tuple
        A tuple containing:
        - images_tensor: [B, 3, H, W] tensor of stacked images
        - targets_batch: list of scaled target annotations for each image in the batch
    """

    images = []
    targets_batch = []
    
    for img, targets in batch:

        images.append(img)
        
        # Scale bbox coordinates: original -> resized image tensor size.
        scaled_targets = []

        # Current resized image size (H, W)
        img_h, img_w = img.shape[-2], img.shape[-1]

        for ann in targets:

            ann_copy = ann.copy()
            
            # bbox format: [x, y, w, h] in original image coordinates.
            # We need the original image size to compute the scale.
            if "original_size" in ann:
                orig_h, orig_w = ann["original_size"]
                scale_x = float(img_w) / float(orig_w)
                scale_y = float(img_h) / float(orig_h)
                
                # Scale from original size to resized image (still in corner format).
                x, y, w, h = ann["bbox"]
                x_scaled = x * scale_x
                y_scaled = y * scale_y
                w_scaled = w * scale_x
                h_scaled = h * scale_y
                
                # Convert from corner format [x, y, w, h] to center format [cx, cy, w, h].
                cx = x_scaled + w_scaled / 2.0
                cy = y_scaled + h_scaled / 2.0
                
                ann_copy["bbox"] = [cx, cy, w_scaled, h_scaled]
                ann_copy["resized_size"] = (img_h, img_w)
            
            scaled_targets.append(ann_copy)
        
        targets_batch.append(scaled_targets)

    # Stack images into a batch tensor
    images_tensor = torch.stack(images, dim=0)  # [B, 3, H, W]
    
    return images_tensor, targets_batch


def get_coco_data(config: dict) -> tuple:
    """
    Builds train/validation COCO datasets and dataloaders.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    tuple
        ``(dataloader_tr, dataloader_val)``.
        Validation entry is ``None`` when ``validation_monitor.enabled`` is false.
    """

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    val_cfg = config.get("validation_monitor", {})
    val_enabled = bool(val_cfg.get("enabled", True))

    # Apply backbone specific image transforms.
    transform = build_transform(config)

    dataset_tr = _build_coco_dataset(
        data_dir=data_cfg["train_data_dir"],
        ann_file=data_cfg["train_ann_file"],
        transform=transform,
    )

    dataset_val = None
    dataloader_val = None

    if val_enabled:
        dataset_val_full = _build_coco_dataset(
            data_dir=data_cfg["val_data_dir"],
            ann_file=data_cfg["val_ann_file"],
            transform=transform,
        )

        # Create subset, if specified in the configuration.
        dataset_val = _build_val_subset(dataset_val_full, val_cfg)
    else:
        print("  Validation monitor disabled: skipping validation dataset/dataloader creation.")

    # Shared loader defaults (train section), with optional validation overrides.
    train_num_workers = int(data_cfg.get("num_workers", 0))
    train_pin_memory = bool(data_cfg.get("pin_memory", True))
    train_persistent_workers = bool(data_cfg.get("persistent_workers", train_num_workers > 0))
    train_prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    val_num_workers = train_num_workers
    val_pin_memory = train_pin_memory
    val_persistent_workers = train_persistent_workers
    val_prefetch_factor = train_prefetch_factor

    train_batch_size = int(train_cfg["batch_size"])
    val_batch_size = int(val_cfg.get("batch_size", train_batch_size))

    dataloader_tr = _build_dataloader(
        dataset=dataset_tr,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=train_pin_memory,
        persistent_workers=train_persistent_workers,
        prefetch_factor=train_prefetch_factor,
        name="Train",
    )

    if val_enabled and dataset_val is not None:
        dataloader_val = _build_dataloader(
            dataset=dataset_val,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
            persistent_workers=val_persistent_workers,
            prefetch_factor=val_prefetch_factor,
            name="Validation",
        )

    return dataloader_tr, dataloader_val


def _build_coco_dataset(
        data_dir: str,
        ann_file: str,
        transform: T.Compose
    ) -> CocoDetection:
    """
    Creates a COCO dataset.
    The original image sizes are attached to annotations.

    Parameters
    ----------
    data_dir : str
        Directory containing COCO images.
    ann_file : str
        Path to COCO annotation JSON file.
    transform : T.Compose
        A torchvision transform pipeline to apply to images.

    Returns
    -------
    CocoDetection
        A COCO dataset with attached original image sizes in annotations.
    """

    dataset = CocoDetection(root=data_dir, annFile=ann_file, transform=transform)

    # Store original image sizes in annotations for bbox scaling.
    coco_api = COCO(ann_file)
    for img_id in coco_api.imgs:
        img_info = coco_api.imgs[img_id]
        for ann_id in coco_api.getAnnIds(imgIds=img_id):
            ann = coco_api.anns[ann_id]
            ann["original_size"] = (img_info["height"], img_info["width"])

    # Update dataset's coco object with original sizes.
    dataset.coco = coco_api
    return dataset


def _build_val_subset(
        dataset_val: CocoDetection, val_cfg: dict
    ) -> Subset | CocoDetection:
    """
    Constructs a validation subset from config.
    The subset will contain a contiguous block of images starting from `start_idx` and
    containing `num_images` images. If the specified range exceeds the dataset size,
    it will be truncated accordingly.

    Parameters
    ----------
    dataset_val : CocoDetection
        The full validation dataset.
    val_cfg : dict
        Validation configuration containing `start_idx` and `num_images` keys.

    Returns
    -------
    Subset or CocoDetection
        A subset of the validation dataset based on the specified range, or the full
        dataset if the range is invalid.
    """

    start_idx = int(val_cfg.get("start_idx", 0))
    num_images = int(val_cfg.get("num_images", 200))
    max_len = len(dataset_val)
    start_idx = max(0, min(start_idx, max_len))
    end_idx = max(start_idx, min(start_idx + max(0, num_images), max_len))

    if end_idx > start_idx:
        indices = list(range(start_idx, end_idx))
        return Subset(dataset_val, indices)
    
    return dataset_val


def _build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    name: str,
) -> DataLoader:
    """Creates a torch DataLoader."""

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "collate_fn": collate_fn_coco,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        dataloader_kwargs["persistent_workers"] = bool(persistent_workers)
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    print(
        f"  {name} DataLoader settings: \n"
        f"    dataset_size={len(dataset)}\n"
        f"    batch_size={dataloader_kwargs['batch_size']}\n"
        f"    num_workers={int(num_workers)}\n"
        f"    pin_memory={bool(pin_memory)}\n"
        f"    persistent_workers={bool(persistent_workers) if int(num_workers) > 0 else False}\n"
        f"    prefetch_factor={int(prefetch_factor) if int(num_workers) > 0 else 'n/a'}\n"
    )

    return DataLoader(**dataloader_kwargs)
