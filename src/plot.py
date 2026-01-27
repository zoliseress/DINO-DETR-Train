
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_losses(csv_file: Path):
    """
    Plot training losses from the given metrics.csv file.
    Aggregates values by epoch (takes mean of each epoch).
    
    Args:
        csv_file: Path to the metrics.csv file
    """

    print(f"Loading metrics from: {csv_file}\n")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Display column names and first few rows
    print("Columns:", df.columns.tolist())
    print("Data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Aggregate by epoch (take mean of all steps within each epoch)
    df_epoch = df.groupby('epoch').mean(numeric_only=True)
    
    print("\nAggregated by epoch:")
    print(df_epoch)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Total loss
    if 'train_loss' in df_epoch.columns:
        axes[0].plot(df_epoch.index, df_epoch['train_loss'], linewidth=2, color='blue', marker='o', markersize=6)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Training Loss (per Epoch)')
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: BBox loss
    if 'train_loss_bbox' in df_epoch.columns:
        axes[1].plot(df_epoch.index, df_epoch['train_loss_bbox'], linewidth=2, color='green', marker='s', markersize=6)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('BBox Loss (L1 + GIoU) (per Epoch)')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Classification loss
    if 'train_loss_cls' in df_epoch.columns:
        axes[2].plot(df_epoch.index, df_epoch['train_loss_cls'], linewidth=2, color='red', marker='^', markersize=6)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Classification Loss (CrossEntropy) (per Epoch)')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(csv_file.parent / 'losses.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'losses.png'")
    plt.show()


def show_image_with_boxes(img_tensor, boxes_px, labels=None, scores=None, img_size=None, 
                          gt_boxes_px=None, gt_labels=None, show_gt=True):
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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
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


if __name__ == "__main__":

    plot_training_losses(Path("lightning_logs/version_30/metrics.csv"))
