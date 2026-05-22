import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou as tv_generalized_box_iou


def generalized_box_iou(boxes1, boxes2):
    """
    Compute GIoU between two sets of boxes (in [cx, cy, w, h] format).
    
    Args:
        boxes1: [N, 4] in [cx, cy, w, h]
        boxes2: [M, 4] in [cx, cy, w, h]
    
    Returns:
        giou: [N, M] GIoU values
    """
    # Convert center format to corner format [x0, y0, x1, y1].
    cx1, cy1, w1, h1 = boxes1.unbind(-1)
    boxes1_xyxy = torch.stack((cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2), dim=-1)

    cx2, cy2, w2, h2 = boxes2.unbind(-1)
    boxes2_xyxy = torch.stack((cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2), dim=-1)

    n, m = boxes1_xyxy.shape[0], boxes2_xyxy.shape[0]
    if n == 0 or m == 0:
        return boxes1.new_zeros((n, m))

    return tv_generalized_box_iou(boxes1_xyxy, boxes2_xyxy)


def hungarian_matching(preds_logits, preds_boxes, targets, device):
    """
    Match predictions to ground truth using Hungarian algorithm.
    Uses Facebook DETR cost weighting: cost_class=1, cost_bbox=5.0, cost_giou=2.0
    
    Args:
        preds_logits: [batch_size, num_queries, num_classes]
        preds_boxes: [batch_size, num_queries, 4] in [cx, cy, w, h] format normalized
        targets: list of COCO-style annotations per image (center format)
        device: torch device
    
    Returns:
        matched_indices: list of (pred_idx, gt_idx) tuples per image
    """
    batch_size = preds_logits.shape[0]
    matched_indices = []
    
    for b in range(batch_size):

        num_gts = len(targets[b])
        
        if num_gts == 0:
            # No ground truth objects
            matched_indices.append([])
            continue
        
        # Get class predictions and boxes for this image
        logits = preds_logits[b].float()  # [num_queries, num_classes]
        boxes = preds_boxes[b].float()    # [num_queries, 4] in [cx, cy, w, h]
        
        # Classification cost: negative probability of correct class
        probs = F.softmax(logits, dim=-1)  # [num_queries, num_classes]
        gt_classes = torch.as_tensor(
            [int(gt["category_id"]) for gt in targets[b]],
            device=device,
            dtype=torch.long,
        )

        gt_boxes = torch.as_tensor(
            [gt["bbox"] for gt in targets[b]],
            device=device,
            dtype=boxes.dtype,
        )

        # GT bbox is in center format [cx, cy, w, h], normalize to [0,1].
        gt_scales = []
        for gt in targets[b]:
            resized_size = gt.get("resized_size", None)
            if resized_size is not None:
                img_h, img_w = resized_size
                gt_scales.append([float(img_w), float(img_h), float(img_w), float(img_h)])
            else:
                gt_scales.append([518.0, 518.0, 518.0, 518.0])

        scale = torch.as_tensor(gt_scales, device=device, dtype=boxes.dtype)
        gt_boxes_norm = gt_boxes / scale

        # Class cost: -prob of correct class (official DETR matcher style).
        class_costs = -probs[:, gt_classes].transpose(0, 1)  # [num_gts, num_queries]

        # BBox cost: L1 distance [num_gts, num_queries].
        bbox_costs = torch.cdist(gt_boxes_norm, boxes, p=1)

        # GIoU cost: 1 - GIoU (convert to cost, lower is better) [num_gts, num_queries].
        giou_vals = generalized_box_iou(boxes, gt_boxes_norm)  # [num_queries, num_gts]
        giou_costs = (1 - giou_vals).transpose(0, 1)
        
        # Combined cost: Facebook DETR weighting
        # cost = cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
        cost_matrix = class_costs + 5.0 * bbox_costs + 2.0 * giou_costs  # [num_gts, num_queries]

        # Check for NaN/Inf and replace with large values
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)

        cost_matrix = cost_matrix.cpu().detach().numpy()
        
        # Check again after conversion
        if np.isnan(cost_matrix).any() or np.isinf(cost_matrix).any():
            print(f"Warning: Invalid values in cost matrix at batch {b}")
            # Skip matching for this batch
            matched_indices.append([])
            continue

        # Hungarian algorithm: find optimal assignment
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        matched_indices.append(list(zip(pred_indices, gt_indices)))
    
    return matched_indices


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_perfect_match():
    """Test when predictions perfectly match ground truth."""
    
    print("\n=== Test 1: Perfect Match ===")
    device = torch.device("cpu")
    
    num_queries = 5
    num_classes = 92
    batch_size = 1
    
    # Create predictions that perfectly match GT
    preds_logits = torch.zeros(batch_size, num_queries, num_classes)
    preds_logits[0, 0, 1] = 10.0  # High confidence for class 1 at query 0
    preds_logits[0, 1, 2] = 10.0  # High confidence for class 2 at query 1
    
    preds_boxes = torch.zeros(batch_size, num_queries, 4)
    preds_boxes[0, 0] = torch.tensor([0.25, 0.25, 0.1, 0.1])  # [cx, cy, w, h]
    preds_boxes[0, 1] = torch.tensor([0.75, 0.75, 0.2, 0.2])
    
    # Create GT targets
    targets = [[
        {"category_id": 1, "bbox": [0.25, 0.25, 0.1, 0.1]},
        {"category_id": 2, "bbox": [0.75, 0.75, 0.2, 0.2]}
    ]]
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    print(f"Matched indices: {matched_indices}")
    assert len(matched_indices) == 1, "Should have 1 batch"
    assert len(matched_indices[0]) == 2, "Should match 2 pairs"
    print("✓ Perfect match test passed!")


def test_empty_targets():
    """Test when there are no ground truth objects."""
    print("\n=== Test 2: Empty Targets ===")
    device = torch.device("cpu")
    
    num_queries = 5
    num_classes = 92
    batch_size = 1
    
    preds_logits = torch.randn(batch_size, num_queries, num_classes)
    preds_boxes = torch.rand(batch_size, num_queries, 4) * 0.5 + 0.25
    
    targets = [[]]  # No ground truth objects
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    print(f"Matched indices: {matched_indices}")
    assert len(matched_indices) == 1, "Should have 1 batch"
    assert len(matched_indices[0]) == 0, "Should have 0 matches"
    print("✓ Empty targets test passed!")


def test_more_queries_than_gt():
    """Test when there are more queries than ground truth objects."""
    print("\n=== Test 3: More Queries Than GT ===")
    device = torch.device("cpu")
    
    num_queries = 10
    num_classes = 92
    batch_size = 1
    
    # Create random predictions
    preds_logits = torch.randn(batch_size, num_queries, num_classes)
    preds_boxes = torch.rand(batch_size, num_queries, 4) * 0.5 + 0.25
    
    # Only 3 GT objects
    targets = [[
        {"category_id": 1, "bbox": [0.2, 0.2, 0.1, 0.1]},
        {"category_id": 5, "bbox": [0.5, 0.5, 0.15, 0.15]},
        {"category_id": 10, "bbox": [0.8, 0.8, 0.12, 0.12]}
    ]]
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    print(f"Matched indices: {matched_indices}")
    assert len(matched_indices) == 1, "Should have 1 batch"
    assert len(matched_indices[0]) == 3, "Should match 3 pairs (limited by GT count)"
    print("✓ More queries than GT test passed!")


def test_multiple_batches():
    """Test with multiple images in batch."""
    print("\n=== Test 4: Multiple Batches ===")
    device = torch.device("cpu")
    
    num_queries = 5
    num_classes = 92
    batch_size = 3
    
    preds_logits = torch.randn(batch_size, num_queries, num_classes)
    preds_boxes = torch.rand(batch_size, num_queries, 4) * 0.5 + 0.25
    
    # Different number of GT objects per image
    targets = [
        [{"category_id": 1, "bbox": [0.25, 0.25, 0.1, 0.1]}],
        [
            {"category_id": 2, "bbox": [0.3, 0.3, 0.1, 0.1]},
            {"category_id": 3, "bbox": [0.7, 0.7, 0.1, 0.1]}
        ],
        []  # No objects in third image
    ]
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    print(f"Matched indices per batch: {[len(m) for m in matched_indices]}")
    assert len(matched_indices) == 3, "Should have 3 batches"
    assert len(matched_indices[0]) == 1, "First batch should have 1 match"
    assert len(matched_indices[1]) == 2, "Second batch should have 2 matches"
    assert len(matched_indices[2]) == 0, "Third batch should have 0 matches"
    print("✓ Multiple batches test passed!")


def test_no_good_matches():
    """Test when predictions are very different from GT."""
    print("\n=== Test 5: No Good Matches (Hard Assignment) ===")
    device = torch.device("cpu")
    
    num_queries = 5
    num_classes = 92
    batch_size = 1
    
    # Create predictions with wrong classes
    preds_logits = torch.zeros(batch_size, num_queries, num_classes)
    preds_logits[0, :, 50] = 10.0  # All queries predict class 50
    
    # Create boxes far from predictions
    preds_boxes = torch.zeros(batch_size, num_queries, 4)
    preds_boxes[0, :] = torch.tensor([0.1, 0.1, 0.05, 0.05])
    
    # Create GT with different classes and locations
    targets = [[
        {"category_id": 1, "bbox": [0.9, 0.9, 0.05, 0.05]},
        {"category_id": 2, "bbox": [0.5, 0.5, 0.1, 0.1]}
    ]]
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    print(f"Matched indices: {matched_indices}")
    assert len(matched_indices) == 1, "Should have 1 batch"
    assert len(matched_indices[0]) == 2, "Should still match 2 pairs (Hungarian finds best assignment)"
    print("✓ No good matches test passed!")


def test_bbox_format():
    """Test that center format [cx, cy, w, h] is correctly handled."""
    print("\n=== Test 6: BBox Format (Center Format) ===")
    device = torch.device("cpu")
    
    num_queries = 2
    num_classes = 92
    batch_size = 1
    
    # Create predictions
    preds_logits = torch.zeros(batch_size, num_queries, num_classes)
    preds_logits[0, 0, 5] = 10.0
    preds_logits[0, 1, 6] = 10.0
    
    # Center format: [cx=0.5, cy=0.5, w=0.2, h=0.2]
    # This represents a box from (0.4, 0.4) to (0.6, 0.6)
    preds_boxes = torch.tensor([
        [[0.5, 0.5, 0.2, 0.2]],
        [[0.3, 0.3, 0.1, 0.1]]
    ]).reshape(batch_size, num_queries, 4)
    
    # GT in center format (will be scaled by 518)
    targets = [[
        {"category_id": 5, "bbox": [259.0, 259.0, 103.6, 103.6]},  # center: 0.5,0.5 w,h: 0.2,0.2
        {"category_id": 6, "bbox": [155.4, 155.4, 51.8, 51.8]}      # center: 0.3,0.3 w,h: 0.1,0.1
    ]]
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    print(f"Matched indices: {matched_indices}")
    assert len(matched_indices) == 1, "Should have 1 batch"
    assert len(matched_indices[0]) == 2, "Should match 2 pairs"
    print("✓ BBox format test passed!")


def test_output_format():
    """Test the output format of matched indices."""
    print("\n=== Test 7: Output Format ===")
    device = torch.device("cpu")
    
    num_queries = 5
    num_classes = 92
    batch_size = 2
    
    preds_logits = torch.randn(batch_size, num_queries, num_classes)
    preds_boxes = torch.rand(batch_size, num_queries, 4) * 0.5 + 0.25
    
    targets = [
        [{"category_id": 1, "bbox": [0.25, 0.25, 0.1, 0.1]}],
        [
            {"category_id": 2, "bbox": [0.3, 0.3, 0.1, 0.1]},
            {"category_id": 3, "bbox": [0.7, 0.7, 0.1, 0.1]}
        ]
    ]
    
    matched_indices = hungarian_matching(preds_logits, preds_boxes, targets, device)
    
    # Check output format
    assert isinstance(matched_indices, list), "Output should be a list"
    assert len(matched_indices) == batch_size, "Should have one entry per batch"
    
    for batch_matches in matched_indices:
        assert isinstance(batch_matches, list), "Each batch should contain a list"
        for pred_idx, gt_idx in batch_matches:
            assert isinstance(pred_idx, (int, np.integer)), "Pred index should be int"
            assert isinstance(gt_idx, (int, np.integer)), "GT index should be int"
    
    print(f"Output format correct: {matched_indices}")
    print("✓ Output format test passed!")


def run_all_tests():
    """Run all test functions."""

    print("=" * 70)
    print("Running Hungarian Matching Tests")
    print("=" * 70)
    
    try:
        test_perfect_match()
        test_empty_targets()
        test_more_queries_than_gt()
        test_multiple_batches()
        test_no_good_matches()
        test_bbox_format()
        test_output_format()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
