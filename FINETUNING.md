# Fine-tuning Guide for DETR

This guide explains how to use the extended training code to fine-tune an existing checkpoint.

## Overview

The training code supports:
- Loading existing checkpoints for fine-tuning
- Freezing/unfreezing the backbone selectively
- Progressive unfreezing (unfreeze backbone at a specific epoch)
- All configuration through `configs/default.yaml`

## Quick Start

To fine-tune a pretrained checkpoint, simply edit `configs/default.yaml`:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/best_epoch=6-step=51751-train_loss=2.9145.ckpt" (insert yours)
  freeze_backbone: true
  unfreeze_backbone_at_epoch: -1
  epochs: 10
```

Then run:
```bash
python src/train.py
```

## Configuration Parameters

All fine-tuning is controlled through `configs/default.yaml`. The relevant parameters are:

```yaml
train:
  checkpoint_path: null  # Path to checkpoint for fine-tuning (null = train from scratch)
  freeze_backbone: true  # Freeze backbone during training
  unfreeze_backbone_at_epoch: -1  # Epoch to unfreeze backbone (-1 = never unfreeze)
  epochs: 20
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_workers: 4
```

**Note:** The only command-line argument is `--config` for specifying a different config file:
```bash
python src/train.py --config configs/my_custom_config.yaml
```

## Fine-tuning Strategies

### 1. Fine-tune with Frozen Backbone

Fine-tune only the DETR head layers while keeping the backbone frozen:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/best_epoch=6-step=51751-train_loss=2.9145.ckpt"
  freeze_backbone: true
  unfreeze_backbone_at_epoch: -1
  epochs: 10
```

This approach:
- Keeps the backbone weights fixed, only trains the DETR head and transformer layers
- Uses less GPU memory and converges faster
- Works well for small datasets

### 2. Fine-tune with Full Backbone Unfreezing

Unfreeze and train all parameters including the backbone:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/best_epoch=6-step=51751-train_loss=2.9145.ckpt"
  freeze_backbone: false
  unfreeze_backbone_at_epoch: -1
  epochs: 20
```

This approach:
- Trains all model parameters
- Requires more GPU memory and takes longer to converge
- Better for larger datasets
- Can lead to better final performance

### 3. Progressive Unfreezing

Start with a frozen backbone and unfreeze it at a later epoch:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/best_epoch=6-step=51751-train_loss=2.9145.ckpt"
  freeze_backbone: true
  unfreeze_backbone_at_epoch: 5
  epochs: 15
```

This approach:
- Freezes backbone initially (fast convergence on head)
- Unfreezes backbone at epoch 5 (fine-tunes backbone)
- Combines benefits of both approaches
- Reduces overfitting risk on small datasets

## Parameter Relationships

`freeze_backbone` and `unfreeze_backbone_at_epoch` parameters work together:

- `freeze_backbone: true` - Initial state when training starts
- `unfreeze_backbone_at_epoch: -1` - Never unfreeze (stays frozen throughout)
- `unfreeze_backbone_at_epoch: N` (where N ≥ 0) - Unfreeze at epoch N (overrides initial freeze)

Example progression with progressive unfreezing:
- Epochs 0-4: Backbone frozen, only DETR head training
- Epoch 5+: Backbone unfrozen, full model fine-tuning

## Available Checkpoints

By default the trained checkpoints are located in:
```
lightning_logs/version_0/checkpoints/
```

Examples:
- `best_epoch=6-step=51751-train_loss=2.9145.ckpt` - Best checkpoint at epoch 6
- `last.ckpt` - Most recent checkpoint

## Configuration Examples

### Example 1: Quick Fine-tune with Frozen Backbone
```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/last.ckpt"
  freeze_backbone: true
  unfreeze_backbone_at_epoch: -1
  epochs: 20
  learning_rate: 0.0001
```

### Example 2: Full Fine-tuning
```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/last.ckpt"
  freeze_backbone: false
  unfreeze_backbone_at_epoch: -1
  epochs: 20
  learning_rate: 0.00005  # Lower LR for full fine-tuning
```

### Example 3: Progressive Unfreezing
```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/last.ckpt"
  freeze_backbone: true
  unfreeze_backbone_at_epoch: 10
  epochs: 20
  learning_rate: 0.0001
```

### Example 4: Training from Scratch (No Checkpoint)
```yaml
train:
  checkpoint_path: null  # No checkpoint, train from scratch
  freeze_backbone: true
  unfreeze_backbone_at_epoch: -1
  epochs: 20
```

## Notes on fine-tuning

1. **Adjust learning rate**: When fine-tuning, consider using a lower learning rate than initial training:
   ```yaml
   train:
     learning_rate: 0.00005  # Lower LR for fine-tuning
   ```

2. **Batch size**: You can fine-tune with a different batch size if needed:
   ```yaml
   train:
     batch_size: 8  # Smaller for limited GPU memory
   ```

3. **Progressive unfreezing strategy**:
   - Small dataset (< 1000 images): Freeze backbone throughout
   - Medium dataset (1000-10000 images): Unfreeze at epoch 5-10
   - Large dataset (> 10000 images): Unfreeze from the start or `freeze_backbone: false`
