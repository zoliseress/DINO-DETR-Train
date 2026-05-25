import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def show_image_with_boxes(
        img_tensor: torch.Tensor,
        boxes_px: torch.Tensor,
        labels: list[str] = None,
        scores: torch.Tensor = None,
        img_size: tuple[int, int] = None, 
        gt_boxes_px: torch.Tensor = None,
        gt_labels: list[str] = None,
        show_gt: bool = True
    ) -> None:
    """
    Visualize image with bounding boxes.
    
    Args:
        img_tensor: [C, H, W] image tensor
        boxes_px: [N, 4] predicted boxes in (x0, y0, x1, y1) pixel coordinates
        labels: [N] predicted class labels
        scores: [N] confidence scores
        img_size: (orig_h, orig_w) original image size for reference
        gt_boxes_px: [M, 4] ground truth boxes in (x0, y0, x1, y1) pixel coordinates (optional)
        gt_labels: [M] ground truth class labels (optional)
        show_gt: whether to display ground truth boxes (default: True)
    """
    
    # Convert to numpy and permute to HWC
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # If image is in [0, 1] range, keep it; if [0, 255], normalize
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_np)
    
    # Draw ground truth boxes (red)
    if show_gt and gt_boxes_px is not None:
        for i, box in enumerate(gt_boxes_px):
            x0, y0, x1, y1 = box
            w = x1 - x0
            h = y1 - y0
            
            rect = patches.Rectangle((x0, y0), w, h, linewidth=2.5, edgecolor='red', 
                                     facecolor='none', linestyle='--', label='GT' if i == 0 else '')
            ax.add_patch(rect)
            
            # Add GT label
            if gt_labels is not None:
                label = gt_labels[i]
                ax.text(x0, y0 - 8, f"GT: {label}", color='red', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Draw predicted boxes (green)
    for i, box in enumerate(boxes_px):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='green', 
                                facecolor='none', label='Pred' if i == 0 else '')
        ax.add_patch(rect)
        
        # Add label and score
        if labels is not None and scores is not None:
            label = labels[i]
            score = scores[i]
            ax.text(x0, y0 - 20, f"{label}: {score:.2f}", color='green', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        elif labels is not None:
            label = labels[i]
            ax.text(x0, y0 - 20, f"{label}", color='green', fontsize=10)
    
    title = "Detections"
    if img_size:
        title += f" (original: {img_size[1]}x{img_size[0]})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    if show_gt and gt_boxes_px is not None:
        ax.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.show()
