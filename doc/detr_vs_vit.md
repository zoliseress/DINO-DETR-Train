# DETR vs Vision Transformer (ViT)
Understanding how DETR uses transformers differently from a Vision Transformer.

## High-Level Idea
- **DETR** uses transformers for **object detection**.
- **ViT** uses transformers for **image classification** or general feature extraction.

Both rely on transformer blocks, but they use them in *different ways*.

---

## 1. Input Representation

### Vision Transformer (ViT)
- Splits the image into fixed-size **patches** (e.g., 16×16).
- Flattens each patch and applies a linear projection → patch embeddings.
- Adds positional encodings.
- Uses a transformer **encoder only**.

### DETR
- Originally uses a **CNN backbone** (e.g., ResNet-50) to extract a feature map. Here the backbone is the more advanced DINOv2.
- Flattens spatial features into a sequence.
- Adds 2D positional encodings.
- Uses **both encoder and decoder**.

ViT replaces the CNN with transformers; DETR adds transformers on top of a CNN (or another feature extractor like DINOv2).

---

## 2. Architecture Differences

### ViT: Encoder-Only
- Processes sequence of patch embeddings.
- Produces a single global “CLS” token for classification.

### DETR: Encoder + Decoder
- **Encoder** processes image features.
- **Decoder** uses **object queries** that attend to the encoder features.
- Each decoder output becomes one predicted object.

---

## 3. Object Queries (DETR’s key innovation)
- A fixed set of learned embeddings (usually more than the number of the objects in the image, e.g. 100).
- Each query attends to image features via cross-attention.
- Each query → one bounding box + class prediction.
- No proposals, anchors, or NMS.

This mechanism does not exist in ViT.

---

## 4. Set Prediction and Hungarian Matching
DETR frames detection as a **set prediction** problem:
- Predictions are matched to ground-truth boxes with the **Hungarian algorithm**.
- Loss includes classification (cross entropy) + bounding box regression (L1 + GIoU).
(- The authors used auxiliary losses from the decoder transformer layers.)

ViT has no concept of matching, boxes, or set prediction.

---

## 5. Output Differences

### ViT
- Single output embedding (CLS token)
- Typically used for classification

### DETR
- Outputs a **set** of:
  - class probabilities
  - bounding boxes (cx, cy, w, h)
  - “no object” class for empty queries

---

## 6. Training Differences

### ViT
- Standard supervised classification
- Loss: cross-entropy

### DETR
- Multi-task detection loss:
  - Hungarian matching loss
  - L1 loss for box coordinates
  - GIoU loss
  - Classification loss

DETR is more computationally demanding to train.

---

## Summary Table

| Feature | ViT | DETR |
|--------|-----|------|
| Task | Classification | Object detection |
| Architecture | Encoder only | Encoder + Decoder |
| Input | Image patches | CNN feature map |
| Object queries | ❌ | ✅ |
| Output | Class label | Set of boxes + classes |
| Loss | Cross-entropy | Classification + L1 + GIoU + matching |
| Pretraining | ImageNet | CNN backbone pretrained |

---

## Short Summary
ViT uses transformers to classify an image using patch embeddings, while DETR uses an encoder–decoder transformer with learned object queries to detect objects as a set, removing the need for anchors and NMS.
