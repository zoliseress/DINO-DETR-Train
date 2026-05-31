# Fine-tuning Guide for DINO-DETR

This guide explains how to use the extended training code to fine-tune an existing DINO-DETR checkpoint.

## Overview

The training code supports:
- Loading existing checkpoints for fine-tuning
- Freezing/unfreezing the backbone selectively
- Progressive unfreezing (unfreeze backbone at a specific epoch)
- Separate backbone learning rate

All configuration through `configs/default.yaml`

## Quick Start

To fine-tune a checkpoint, simply edit `configs/default.yaml` or your custom config yaml file:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/XXX.ckpt"
  freeze_backbone: true
  unfreeze_backbone_at_epoch: -1
  learning_rate: 0.0001
  backbone_learning_rate: 0.00001
  epochs: 10
```

Then run:
```bash
python src/train.py
```
or
```bash
python src/train.py --config configs/my_custom_config.yaml
```

## Fine-tuning Strategies

### 1. Fine-tune with frozen backbone (recommended for small datasets)

Fine-tune only the DETR head layers while keeping the backbone frozen:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/XXX.ckpt"
  freeze_backbone: true
  unfreeze_backbone_at_epoch: -1
  epochs: 10
```

This approach:
- Keeps the DINO backbone weights fixed
- Only trains the DETR head and transformer layers
- Uses less GPU memory
- Converges faster
- Works well for small datasets

### 2. Fine-tune with full backbone unfreezing

Unfreeze and train all parameters including the backbone:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/XXX.ckpt"
  freeze_backbone: false
  unfreeze_backbone_at_epoch: -1
  epochs: 20
```

This approach:
- Trains all model parameters
- Requires more GPU memory
- Takes longer to converge
- Better for larger datasets
- Can lead to better final performance

### 3. Progressive unfreezing (recommended for medium datasets)

Start with a frozen backbone and unfreeze it at a later epoch:

```yaml
train:
  checkpoint_path: "lightning_logs/version_0/checkpoints/XXX.ckpt"
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

### `freeze_backbone` and `unfreeze_backbone_at_epoch`

These parameters work together:

- **`freeze_backbone: true`** - Initial state when training starts
- **`unfreeze_backbone_at_epoch: -1`** - Never unfreeze (stays frozen throughout)
- **`unfreeze_backbone_at_epoch: N`** (where N ≥ 0) - Unfreeze at epoch N (overrides initial freeze)

Example progression with progressive unfreezing:
- Epochs 0-4: Backbone frozen, only DETR head training
- Epoch 5+: Backbone unfrozen, full model fine-tuning

## Available Checkpoints

Your trained checkpoints are located in:
```
lightning_logs/version_0/checkpoints/
```

Examples:
- `best_epoch=14-step=55455-val_loss=11.5621-train_loss=9.9293.ckpt` - Best checkpoint at epoch 14
- `last.ckpt` - Most recent checkpoint
