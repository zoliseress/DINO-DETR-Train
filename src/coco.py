import torch
import torchvision.transforms as T

from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader


def collate_fn_coco(batch):
    """
    Custom collate function that resizes the images and scales the bounding boxes.

    1. DINOv2 was trained with 518x518 images, therefore the images need to be resized
    to this size.
    2. DETR uses bounding boxes in center (cx, cy, w, h) format, so we need to convert
    from corner (x, y, w, h) format to center (cx, cy, w, h) format.
    
    COCO dataset returns (image, targets) tuples where:
    - image: [3, H, W] tensor after transforms
    - targets: list of annotation dicts with bbox in original image coordinates
    """

    images = []
    targets_batch = []
    
    for img, targets in batch:

        images.append(img)
        
        # Scale bbox coordinates: original -> 518x518.
        scaled_targets = []

        for ann in targets:

            ann_copy = ann.copy()
            
            # bbox format: [x, y, w, h] in original image coordinates.
            # We need the original image size to compute the scale.
            if "original_size" in ann:
                orig_h, orig_w = ann["original_size"]
                scale_x = 518.0 / orig_w
                scale_y = 518.0 / orig_h
                
                # Scale from original size to 518x518 (still in corner format).
                x, y, w, h = ann["bbox"]
                x_scaled = x * scale_x
                y_scaled = y * scale_y
                w_scaled = w * scale_x
                h_scaled = h * scale_y
                
                # Convert from corner format [x, y, w, h] to center format [cx, cy, w, h].
                cx = x_scaled + w_scaled / 2.0
                cy = y_scaled + h_scaled / 2.0
                
                ann_copy["bbox"] = [cx, cy, w_scaled, h_scaled]
            
            scaled_targets.append(ann_copy)
        
        targets_batch.append(scaled_targets)

    # Stack images into a batch tensor
    images_tensor = torch.stack(images, dim=0)  # [B, 3, 518, 518]
    
    return images_tensor, targets_batch


def get_coco_data(config: dict):
    """Get COCO dataset and dataloader.
    Only for the training data, validation data is not included.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    CocoDetection, torch.utils.data.DataLoader
        Train dataset and train dataloader.
    """

    data_dir = config["data"]["train_data_dir"]
    ann_file = config["data"]["train_ann_file"]
    # data_dir = config["data"]["val_data_dir"]
    # ann_file = config["data"]["val_ann_file"]

    # For DINOv2 the image size shall be divisible by 14.
    # (It was trained with 518x518 images.)
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
    ])

    dataset_tr = CocoDetection(root=data_dir, annFile=ann_file, transform=transform)

    # Store original image sizes in annotations for bbox scaling.
    coco_api = COCO(ann_file)
    for img_id in coco_api.imgs:
        img_info = coco_api.imgs[img_id]
        for ann_id in coco_api.getAnnIds(imgIds=img_id):
            ann = coco_api.anns[ann_id]
            ann["original_size"] = (img_info["height"], img_info["width"])
    
    # Update dataset's coco object with original sizes.
    dataset_tr.coco = coco_api

    dataloader_tr = DataLoader(
        dataset_tr,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_coco
    )

    return dataset_tr, dataloader_tr
