
import numpy as np
import torch
import torchvision.transforms as T
from lightning_module import DETR_Lightning
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from plot import show_image_with_boxes
from train import load_config


# ---------- Helper functions ----------

def box_cxcywh_to_xyxy(boxes):
    """Convert from center format [cx, cy, w, h] to xyxy format [x0, y0, x1, y1].
    
    Args:
        boxes: [N, 4] in (cx, cy, w, h) normalized coordinates [0, 1]
    Returns:
        [N, 4] in (x0, y0, x1, y1) normalized coordinates
    """
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def rescale_bboxes(boxes_xyxy, size):
    """Rescale bounding boxes from normalized [0,1] to pixel coordinates.
    
    Args:
        boxes_xyxy: [N, 4] in normalized coords (0..1) format [x0, y0, x1, y1]
        size: (H, W) of the original image
    Returns:
        [N, 4] in pixel coordinates
    """
    h, w = size
    boxes = boxes_xyxy.clone()
    boxes[:, 0] *= w  # x0
    boxes[:, 1] *= h  # y0
    boxes[:, 2] *= w  # x1
    boxes[:, 3] *= h  # y1
    return boxes


def load_coco_class_names(ann_file):
    """Load COCO class names from annotation file.
    
    Returns a dict mapping category_id to class name.
    """
    coco = COCO(ann_file)
    class_names = {}
    for cat in coco.cats.values():
        class_names[cat['id']] = cat['name']
    return class_names


# ---------- Main entry point ----------

if __name__ == "__main__":

    print("\n    ==== START INFERENCE ====\n\n")

    # 1. Init model.

    print("  Loading model checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ckpt_path = "lightning_logs/version_34/checkpoints/epoch=16-step=62849.ckpt"  # szar
    # ckpt_path = "lightning_logs/version_35/checkpoints/best_epoch=6-step=51751-train_loss=5.1994.ckpt"
    # ckpt_path = "lightning_logs/version_36/checkpoints/best_epoch=17-step=133074-train_loss=5.2166.ckpt"
    # ckpt_path = "lightning_logs/version_38/checkpoints/best_epoch=13-step=103502-train_loss=2.6767.ckpt"  # clock in big BB
    ckpt_path = "lightning_logs/version_1/checkpoints/best_epoch=4-step=36965-train_loss=2.7194.ckpt"  # ?
    config = load_config("configs/default.yaml")

    model = DETR_Lightning.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    model.to(device)

    # 2. Get data.

    print("  Get data...")
    idx = 0

    # Resizing transform for inference (this shall match the training transform).
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
    ])

    # Load and preprocess image
    original_dataset = CocoDetection(
        root=config["data"]["val_data_dir"],
        annFile=config["data"]["val_ann_file"],
        transform=None  # Get original first
    )

    # Get original size before resizing.
    original_img_pil = original_dataset[idx][0]
    orig_w, orig_h = original_img_pil.size  # (W, H)

    # ann = original_dataset[idx][1]
    # print(ann[0]["bbox"])
    # print(ann[0]["category_id"])

    # Resize for model input
    img_resized = transform(original_img_pil)
    input_tensor = img_resized.unsqueeze(0).to(device)  # [1, 3, 518, 518]
    
    # Load COCO class names for display
    class_names = load_coco_class_names(config["data"]["val_ann_file"])
    
    # 3. Inference.

    print("  Run inference...")

    with torch.no_grad():
        outputs = model(input_tensor)  # outputs: dict with 'pred_logits' and 'pred_boxes'

    pred_logits = outputs["pred_logits"][0]  # [num_queries, num_classes(+1)]
    pred_boxes = outputs["pred_boxes"][0]    # [num_queries, 4] [x, y, w, h] normalized

    # 4. Post-process outputs

    print("  Post-process outputs...")

    # Convert logits -> probabilities (float).
    probs = pred_logits.softmax(-1)  # [num_queries, num_classes+1]

    # Exclude last index (background / no-object) when computing best class & score
    probs_no_bg = probs[:, :-1]              # [num_queries, num_classes]
    scores_no_bg, labels_no_bg = probs_no_bg.max(dim=-1)  # per-query best non-bg score & label

    print("num queries:", probs.shape[0])
    print("num non-zero (score>0):", (scores_no_bg > 0.0).sum().item())

    # Choose top-K predictions by non-background score
    top_k = 10
    topk_scores, topk_indices = torch.topk(scores_no_bg, k=min(top_k, scores_no_bg.shape[0]))
    print("topk_scores:", topk_scores)
    print("topk_indices:", topk_indices)

    # Optionally filter by a minimum score threshold (uncomment and tune if you want)
    score_threshold = 0.05
    keep_mask = topk_scores > score_threshold
    if keep_mask.sum().item() == 0:
        print("No top-k detections above score threshold; relaxing threshold to show top predictions.")
        keep_mask = torch.ones_like(keep_mask, dtype=torch.bool)

    selected_query_idx = topk_indices[keep_mask]        # indices into queries (0..num_queries-1)
    selected_scores = topk_scores[keep_mask].cpu().numpy()
    selected_labels = labels_no_bg[selected_query_idx].cpu().numpy()
    selected_boxes_norm = pred_boxes[selected_query_idx].cpu()  # [k, 4] cx,cy,w,h

    print("selected count:", selected_query_idx.numel())
    print("selected_query_idx:", selected_query_idx)

    # 5. Visualize results.

    print("  Visualize predictions...")

    # Convert to xyxy normalized coords (this function is now correct for center format)
    boxes_xyxy = box_cxcywh_to_xyxy(selected_boxes_norm)

    # Rescale to original image size (currently they are in 518x518).
    boxes_px = rescale_bboxes(boxes_xyxy, (orig_h, orig_w)).numpy()

    # Convert label IDs to class names
    class_label_strs = [
        class_names.get(int(label_id), f"Unknown({label_id})") for label_id in selected_labels
    ]

    ############
    # === Debug prints ===
    if True:
        print("boxes_px shape:", boxes_px.shape)
        print("boxes_px (x0,y0,x1,y1):\n", boxes_px)

        # Inspect normalized boxes too
        print("selected_boxes_norm (cx,cy,w,h):\n", selected_boxes_norm.numpy())

        # Compute some diagnostics
        xs = boxes_px[:, 0]
        ys = boxes_px[:, 1]
        x1s = boxes_px[:, 2]
        y1s = boxes_px[:, 3]
        widths = x1s - xs
        heights = y1s - ys
        areas = widths * heights
        print("widths:", widths)
        print("heights:", heights)
        print("areas:", areas)
        print("min area:", float(areas.min()) if areas.size else None)
        print("max area:", float(areas.max()) if areas.size else None)

        # Check for duplicates (round to 1 px to avoid fp noise)
        import numpy as _np
        uniq = _np.unique(_np.round(boxes_px, 2), axis=0)
        print("unique boxes (rounded):", uniq.shape[0])
        if uniq.shape[0] < boxes_px.shape[0]:
            print("Warning: many selected boxes are identical (or nearly identical).")

        # If boxes are degenerate (very small area), print the indices
        degenerate_idx = _np.where(areas < 1.0)[0]
        if degenerate_idx.size > 0:
            print("Degenerate boxes (area < 1 px) at indices:", degenerate_idx)

        print("\nDetected classes:")
        for i, (label_str, score) in enumerate(zip(class_label_strs, selected_scores)):
            print(f"  {i}: {label_str} (score: {score:.3f})")
    ############
  
    # Load and rescale ground truth boxes.
    gt_anns = original_dataset[idx][1]
    gt_boxes_px = []
    gt_labels_str = []
    
    for ann in gt_anns:
        # COCO format: [x, y, w, h] in original image coordinates
        x, y, w, h = ann["bbox"]
        
        # Convert to xyxy format
        x0, y0, x1, y1 = x, y, x + w, y + h
        gt_boxes_px.append([x0, y0, x1, y1])
        
        # Get class name
        cat_id = ann["category_id"]
        cat_name = class_names.get(cat_id, f"Unknown({cat_id})")
        gt_labels_str.append(cat_name)
    
    gt_boxes_px = np.array(gt_boxes_px) if gt_boxes_px else None

    original_img_tensor = ToTensor()(original_img_pil)

    # Display with original image, predicted boxes (green), and ground truth boxes (red)
    show_image_with_boxes(
        original_img_tensor,
        boxes_px,
        labels=class_label_strs,
        scores=selected_scores,
        img_size=(orig_h, orig_w),
        gt_boxes_px=gt_boxes_px,
        gt_labels=gt_labels_str,
        show_gt=True,  # Set to False to hide ground truth boxes
    )
