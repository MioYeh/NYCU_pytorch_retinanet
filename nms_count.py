#!/usr/bin/env python3
"""
Post-process predicted_detections_global.csv:
  - Apply NMS on base-level coordinates per slide
  - Report per-slide and total STAS count

Usage:
    python nms_count.py --input predicted_detections_global.csv
    python nms_count.py --input predicted_detections_global.csv \
        --iou_threshold 0.3 --score_threshold 0.3 \
        --output nms_result.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import List, Tuple


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms(boxes: List[List[float]], scores: List[float],
        iou_threshold: float) -> List[int]:
    """
    Greedy NMS. Returns indices of kept boxes sorted by score (descending).
    boxes: list of [x1, y1, x2, y2]
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept: List[int] = []

    while order:
        current = order.pop(0)
        kept.append(current)
        order = [
            i for i in order
            if compute_iou(boxes[current], boxes[i]) < iou_threshold
        ]

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description='NMS post-processing for WSI detection CSV.')
    parser.add_argument('--input', required=True,
                        help='Path to predicted_detections_global.csv')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for NMS (default: 0.3)')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Minimum score to keep before NMS (default: 0.0)')
    parser.add_argument('--output', default=None,
                        help='Optional: save NMS-filtered detections to this CSV')
    args = parser.parse_args()

    # ---- read CSV ----
    slide_boxes: dict = defaultdict(list)   # slide -> list of (score, box, row)
    total_raw = 0

    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row['score'])
            if score < args.score_threshold:
                continue
            box = [
                float(row['base_x1']),
                float(row['base_y1']),
                float(row['base_x2']),
                float(row['base_y2']),
            ]
            slide_boxes[row['slide_name']].append((score, box, row))
            total_raw += 1

    print(f'\nInput : {args.input}')
    print(f'score_threshold : {args.score_threshold}')
    print(f'iou_threshold   : {args.iou_threshold}')
    print(f'Total raw boxes : {total_raw}\n')

    # ---- per-slide NMS ----
    kept_rows = []
    print(f'{"Slide":<60} {"Raw":>5} {"After NMS":>10}')
    print('-' * 78)

    total_after = 0
    for slide_name in sorted(slide_boxes.keys()):
        entries = slide_boxes[slide_name]
        scores = [e[0] for e in entries]
        boxes  = [e[1] for e in entries]
        rows   = [e[2] for e in entries]

        kept_indices = nms(boxes, scores, args.iou_threshold)

        raw_count   = len(entries)
        kept_count  = len(kept_indices)
        total_after += kept_count

        # display slide name without extension
        stem = os.path.splitext(slide_name)[0]
        print(f'{stem:<60} {raw_count:>5} {kept_count:>10}')

        for idx in kept_indices:
            kept_rows.append(rows[idx])

    print('-' * 78)
    print(f'{"TOTAL":<60} {total_raw:>5} {total_after:>10}\n')

    # ---- save output ----
    if args.output:
        fieldnames = list(kept_rows[0].keys()) if kept_rows else []
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        print(f'NMS result saved to: {args.output}')


if __name__ == '__main__':
    main()
