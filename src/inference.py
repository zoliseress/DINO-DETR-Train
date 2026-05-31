
import numpy as np
import torch
from lightning_module import DETR_Lightning
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from coco_eval_val2017 import box_cxcywh_to_xyxy
from plot import show_image_with_boxes
from train import load_config
from coco import build_transform


# ---------- Helper functions ----------

def rescale_bboxes(boxes_xyxy: torch.Tensor, size: tuple):
    """
    Rescale bounding boxes from normalized [0,1] to pixel coordinates.
    
    Parameters:
    -----------
    boxes_xyxy: torch.Tensor
        [N, 4] in normalized coords (0..1) format [x0, y0, x1, y1]
    size: tuple
        (H, W) of the original image

    Returns:
    -------
        torch.Tensor
        [N, 4] in pixel coordinates
    """
    h, w = size
    boxes = boxes_xyxy.clone()
    boxes[:, 0] *= w  # x0
    boxes[:, 1] *= h  # y0
    boxes[:, 2] *= w  # x1
    boxes[:, 3] *= h  # y1
    return boxes


def load_coco_class_names(ann_file: str) -> dict[int, str]:
    """Load COCO class names from annotation file.
    
    Parameters:
    -----------
    ann_file: str
        Path to COCO annotation JSON file.
    
    Returns:
    -------
    dict[int, str]
        A dict mapping category_id to class name.
    """
    coco = COCO(ann_file)
    class_names = {}
    for cat in coco.cats.values():
        class_names[cat['id']] = cat['name']
    return class_names


if __name__ == "__main__":

    print("\n    ==== START INFERENCE ====\n\n")

    # 1. Init model.

    print("  Loading model checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "lightning_logs/version_9/checkpoints/last.ckpt"
    config = load_config("lightning_logs/version_9/config.yaml")

    model = DETR_Lightning.load_from_checkpoint(ckpt_path, config=config, weights_only=False)
    model.eval()
    model.to(device)

    # 2. Get data.

    print("  Get data...")
    idx = 5  # person + ski
    # idx = 10  # person, boat, bird, handbag
    # idx = 13  # sandwich + bowl

    # Post-processing presets for visualization.
    # - clean: fewer boxes, higher precision
    # - debug: more boxes, higher recall
    viz_mode = "clean"  # "clean" | "debug"
    postprocess_presets = {
        "clean": {
            "top_k": 100,
            "score_threshold": 0.25,
            "bg_prob_threshold": 0.50,
            "margin_threshold": 0.20,
            "max_detections": 12,
        },
        "debug": {
            "top_k": 100,
            "score_threshold": 0.10,
            "bg_prob_threshold": 0.65,
            "margin_threshold": 0.05,
            "max_detections": 25,
        },
    }
    if viz_mode not in postprocess_presets:
        raise ValueError(f"Unsupported viz_mode={viz_mode}. Use one of: {list(postprocess_presets)}")

    post_cfg = postprocess_presets[viz_mode]
    top_k = int(post_cfg["top_k"])
    score_threshold = float(post_cfg["score_threshold"])
    bg_prob_threshold = float(post_cfg["bg_prob_threshold"])
    margin_threshold = float(post_cfg["margin_threshold"])
    max_detections = int(post_cfg["max_detections"])

    # Resizing/normalization transform for inference (must match training transform).
    transform = build_transform(config)

    # Load and preprocess image.
    original_dataset = CocoDetection(
        root=config["data"]["val_data_dir"],
        annFile=config["data"]["val_ann_file"],
        transform=None
    )

    # Load COCO class names for display
    class_names = load_coco_class_names(config["data"]["val_ann_file"])

    # Get original size before resizing.
    original_img_pil = original_dataset[idx][0]
    orig_w, orig_h = original_img_pil.size  # (W, H)

    # Resize for model input.
    img_resized = transform(original_img_pil)
    input_tensor = img_resized.unsqueeze(0).to(device)
    
    # 3. Inference.

    print("\n  Run single-image inference...")

    with torch.no_grad():
        outputs = model(input_tensor)

    pred_logits = outputs["pred_logits"][0]  # [num_queries, num_classes(+1)]
    pred_boxes = outputs["pred_boxes"][0]    # [num_queries, 4] [x, y, w, h] normalized

    # 4. Post-process outputs

    print("  Post-process outputs...")

    # Convert logits -> probabilities (float).
    probs = pred_logits.softmax(-1)  # [num_queries, num_classes+1]
    probs_bg = probs[:, -1]  # Background probability per query

    # Exclude last index (background / no-object) when computing best class & score
    probs_no_bg = probs[:, :-1]  # [num_queries, num_classes]
    scores_no_bg, labels_no_bg = probs_no_bg.max(dim=-1)  # per-query best non-bg score & label

    # Choose top-K predictions by non-background score
    topk_scores, topk_indices = torch.topk(scores_no_bg, k=min(top_k, scores_no_bg.shape[0]))

    # Filter by non-background score and background probability.
    topk_bg_probs = probs_bg[topk_indices]
    topk_margins = topk_scores - topk_bg_probs
    keep_mask = (
        (topk_scores > score_threshold)
        & (topk_bg_probs < bg_prob_threshold)
        & (topk_margins > margin_threshold)
    )
    if keep_mask.sum().item() == 0:
        print(
            "No confident detections passed score/background/margin thresholds: "
            f"score>{score_threshold}, p_bg<{bg_prob_threshold}, margin>{margin_threshold}."
        )

    selected_query_idx = topk_indices[keep_mask]  # indices into queries (0..num_queries-1)
    num_passing = int(selected_query_idx.numel())
    if selected_query_idx.numel() > max_detections:
        selected_query_idx = selected_query_idx[:max_detections]
    num_kept = int(selected_query_idx.numel())

    print(
        f"Post-process summary ({viz_mode}): "
        f"queries={scores_no_bg.shape[0]}, passing={num_passing}, kept_for_viz={num_kept}, "
        f"max_detections={max_detections}"
    )

    selected_scores = scores_no_bg[selected_query_idx].cpu().numpy()
    selected_bg_probs = probs_bg[selected_query_idx].cpu().numpy()
    selected_margins = (scores_no_bg[selected_query_idx] - probs_bg[selected_query_idx]).cpu().numpy()
    selected_labels = labels_no_bg[selected_query_idx].cpu().numpy()
    selected_boxes_norm = pred_boxes[selected_query_idx].cpu()  # [k, 4] cx,cy,w,h

    # 5. Visualize results.

    print("  Visualize predictions...")

    # Convert to xyxy normalized coords (this function is now correct for center format)
    boxes_xyxy = box_cxcywh_to_xyxy(selected_boxes_norm)

    # Rescale to original image size using the resized tensor dimensions.
    resized_h, resized_w = img_resized.shape[-2], img_resized.shape[-1]
    boxes_px_resized = rescale_bboxes(boxes_xyxy, (resized_h, resized_w)).numpy()

    # Map from resized coordinates back to original image coordinates.
    scale_x = orig_w / float(resized_w)
    scale_y = orig_h / float(resized_h)
    boxes_px = boxes_px_resized.copy()
    boxes_px[:, 0] *= scale_x
    boxes_px[:, 2] *= scale_x
    boxes_px[:, 1] *= scale_y
    boxes_px[:, 3] *= scale_y

    if True:
        print(f"Resize mapping: resized=({resized_w}x{resized_h}) -> orig=({orig_w}x{orig_h})")

    # Convert label IDs to class names.
    class_label_strs = [
        class_names.get(int(label_id), f"Unknown({label_id})") for label_id in selected_labels
    ]
  
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

    # Display with original image, predicted boxes (green), and ground truth boxes (red).
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
