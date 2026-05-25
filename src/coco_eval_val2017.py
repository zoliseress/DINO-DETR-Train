import argparse
import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from coco import build_transform
from datamodule import load_config
from lightning_module import DETR_Lightning


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [cx, cy, w, h] to [x0, y0, x1, y1]."""
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


class CocoEvalDataset(Dataset):
    """COCO dataset wrapper that also returns image_id and original image size."""

    def __init__(self, root: str, ann_file: str, transform=None):
        self.dataset = CocoDetection(root=root, annFile=ann_file, transform=None)
        self.transform = transform
        self.to_tensor = ToTensor()
        self.image_ids = list(self.dataset.ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        image_id = int(self.image_ids[index])
        orig_w, orig_h = image.size

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = self.to_tensor(image)

        return image_tensor, image_id, (orig_h, orig_w)


def collate_eval_batch(batch):
    """Stack fixed-size images and keep metadata in Python lists."""

    images = torch.stack([sample[0] for sample in batch], dim=0)
    image_ids = [sample[1] for sample in batch]
    orig_sizes = [sample[2] for sample in batch]
    return images, image_ids, orig_sizes


def build_eval_indices(
        dataset_len: int, start_idx: int, num_images: int
    ) -> list[int]:
    """Return contiguous indices for evaluation."""

    if dataset_len <= 0:
        return []

    start = max(0, min(int(start_idx), dataset_len))
    if int(num_images) <= 0:
        end = dataset_len
    else:
        end = min(dataset_len, start + int(num_images))

    return list(range(start, end))


def predictions_from_batch(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    image_ids: list[int],
    orig_sizes: list[tuple[int, int]],
    valid_cat_ids: set[int],
    score_threshold: float,
    max_detections_per_image: int,
    class_id_offset: int,
) -> list[dict]:
    """Convert model outputs for one batch into COCO detection dictionaries."""

    probs = pred_logits.softmax(-1)
    probs_no_bg = probs[:, :, :-1]
    scores, labels = probs_no_bg.max(dim=-1)

    batch_results = []
    batch_size = pred_logits.shape[0]

    for b in range(batch_size):
        image_id = int(image_ids[b])
        orig_h, orig_w = orig_sizes[b]

        query_scores = scores[b]
        query_labels = labels[b]
        query_boxes = pred_boxes[b]

        top_k = min(int(max_detections_per_image), int(query_scores.shape[0]))
        if top_k <= 0:
            continue

        top_scores, top_indices = torch.topk(query_scores, k=top_k)

        selected_scores = top_scores.detach().cpu().tolist()
        selected_indices = top_indices.detach().cpu().tolist()

        for score, query_idx in zip(selected_scores, selected_indices):
            if float(score) < float(score_threshold):
                continue

            pred_class = int(query_labels[query_idx].item())
            coco_category_id = pred_class + int(class_id_offset)
            if coco_category_id not in valid_cat_ids:
                continue

            box_xyxy = box_cxcywh_to_xyxy(query_boxes[query_idx : query_idx + 1])[0]
            x0, y0, x1, y1 = box_xyxy.detach().cpu().tolist()

            x0 = max(0.0, min(1.0, float(x0)))
            y0 = max(0.0, min(1.0, float(y0)))
            x1 = max(0.0, min(1.0, float(x1)))
            y1 = max(0.0, min(1.0, float(y1)))

            bw = max(0.0, x1 - x0)
            bh = max(0.0, y1 - y0)
            if bw <= 0.0 or bh <= 0.0:
                continue

            batch_results.append(
                {
                    "image_id": image_id,
                    "category_id": coco_category_id,
                    "bbox": [
                        x0 * float(orig_w),
                        y0 * float(orig_h),
                        bw * float(orig_w),
                        bh * float(orig_h),
                    ],
                    "score": float(score),
                }
            )

    return batch_results


def evaluate_coco(
    model: torch.nn.Module,
    dataloader: DataLoader,
    image_ids: list[int],
    coco_gt: COCO,
    device: str,
    score_threshold: float,
    max_detections_per_image: int,
    class_id_offset: int,
) -> tuple[list[dict], dict]:
    """Run model on dataloader and compute COCO bbox metrics."""

    valid_cat_ids = {int(cat_id) for cat_id in coco_gt.getCatIds()}
    detections = []

    model.eval()

    processed = 0
    total = len(image_ids)

    with torch.inference_mode():
        for images, batch_image_ids, batch_orig_sizes in dataloader:
            images = images.to(device)
            outputs = model(images)

            batch_detections = predictions_from_batch(
                pred_logits=outputs["pred_logits"],
                pred_boxes=outputs["pred_boxes"],
                image_ids=batch_image_ids,
                orig_sizes=batch_orig_sizes,
                valid_cat_ids=valid_cat_ids,
                score_threshold=score_threshold,
                max_detections_per_image=max_detections_per_image,
                class_id_offset=class_id_offset,
            )

            detections.extend(batch_detections)
            processed += len(batch_image_ids)

            if processed % 100 == 0 or processed == total:
                print(f"  Processed {processed}/{total} images")

    if not detections:
        raise RuntimeError(
            "No detections were produced. "
            "Try lowering --score-threshold or checking class-id mapping."
        )

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_max1",
        "AR_max10",
        "AR_max100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = {name: float(value) for name, value in zip(metric_names, coco_eval.stats)}

    return detections, metrics


def parse_args():
    """Parse command-line arguments for COCO evaluation."""

    parser = argparse.ArgumentParser(description="Run COCOeval AP on val2017 locally.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file with COCO val paths.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Falls back to train.checkpoint_path from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Eval batch size override (default: validation_monitor.batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers override (default: data.num_workers).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index in val set.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=-1,
        help="Number of images to evaluate. <=0 means full val set.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum score to keep a detection.",
    )
    parser.add_argument(
        "--max-detections-per-image",
        type=int,
        default=100,
        help="Maximum kept predictions per image.",
    )
    parser.add_argument(
        "--class-id-offset",
        type=int,
        default=0,
        help="Applied as: coco_category_id = pred_class + class_id_offset.",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default="outputs/cocoeval_val2017_predictions.json",
        help="Path to save detections JSON in COCO result format.",
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default="outputs/cocoeval_val2017_metrics.json",
        help="Path to save summarized AP/AR metrics JSON.",
    )

    return parser.parse_args()


def resolve_checkpoint_path(config, cli_checkpoint: str | None) -> str:
    """Resolve checkpoint path from CLI or config."""

    if cli_checkpoint:
        return str(cli_checkpoint)

    train_cfg = config.get("train", {})
    checkpoint_path = train_cfg.get("checkpoint_path", None)
    if checkpoint_path:
        return str(checkpoint_path)

    raise ValueError(
        "Checkpoint path not provided. Use --checkpoint or set train.checkpoint_path in config."
    )


def main():

    # Get command line arguments.
    args = parse_args()

    # Load config (the config for the training run is okay).
    config = load_config(args.config)
    data_cfg = config.get("data", {})

    # Get data.
    val_data_dir = str(data_cfg["val_data_dir"])
    val_ann_file = str(data_cfg["val_ann_file"])
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)

    # Print info.
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    batch_size = int(
        args.batch_size or config.get("validation_monitor", {}).get("batch_size", 8)
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0)
    )

    print("\n==== COCOeval val2017 ====")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  device: {device}")
    print(f"  val_data_dir: {val_data_dir}")
    print(f"  val_ann_file: {val_ann_file}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_workers: {num_workers}\n")

    # Build dataset and dataloader for evaluation.
    transform = build_transform(config)
    dataset_full = CocoEvalDataset(
        root=val_data_dir,
        ann_file=val_ann_file,
        transform=transform,
    )

    eval_indices = build_eval_indices(
        dataset_len=len(dataset_full),
        start_idx=args.start_idx,
        num_images=args.num_images,
    )
    if not eval_indices:
        raise RuntimeError("No images selected for evaluation. Check --start-idx and --num-images.")

    eval_image_ids = [int(dataset_full.image_ids[i]) for i in eval_indices]
    dataset_eval = dataset_full if len(eval_indices) == len(dataset_full) else Subset(dataset_full, eval_indices)

    dataloader = DataLoader(
        dataset_eval,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device == "cuda"),
        collate_fn=collate_eval_batch,
    )

    model = DETR_Lightning.load_from_checkpoint(
        checkpoint_path,
        config=config,
        weights_only=False,
    )
    model.to(device)

    coco_gt = COCO(val_ann_file)

    print(f"  evaluating_images: {len(eval_image_ids)}")

    detections, metrics = evaluate_coco(
        model=model,
        dataloader=dataloader,
        image_ids=eval_image_ids,
        coco_gt=coco_gt,
        device=device,
        score_threshold=float(args.score_threshold),
        max_detections_per_image=int(args.max_detections_per_image),
        class_id_offset=int(args.class_id_offset),
    )

    predictions_path = Path(args.save_predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(detections, f)

    metrics_payload = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "num_images": int(len(eval_image_ids)),
        "start_idx": int(args.start_idx),
        "score_threshold": float(args.score_threshold),
        "max_detections_per_image": int(args.max_detections_per_image),
        "class_id_offset": int(args.class_id_offset),
        "metrics": metrics,
    }

    metrics_path = Path(args.save_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved outputs:")
    print(f"  predictions: {predictions_path}")
    print(f"  metrics: {metrics_path}")


if __name__ == "__main__":
    """
    Run COCOeval AP on val2017 locally.
    
    Example usage:
      python src/coco_eval_val2017.py --config config.yaml --checkpoint checkpoint.ckpt

    The config file should specify val_data_dir and val_ann_file under the data section, e.g.:
      data:
        val_data_dir: /path/to/val/images
        val_ann_file: /path/to/val/annotations.json

    The script will save:
        - detections in COCO result format: outputs/cocoeval_val2017_predictions.json
        - summarized metrics (AP/AR): outputs/cocoeval_val2017_metrics.json
    """

    main()
