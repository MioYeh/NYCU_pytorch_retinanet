#!/usr/bin/env python3
"""Draw RetinaNet predicted boxes on saved WSI tile images."""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def load_detections(detections_csv: str) -> Dict[str, List[dict]]:
    rows_by_path: Dict[str, List[dict]] = defaultdict(list)
    with open(detections_csv, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'relative_patch_path', 'score', 'local_x1', 'local_y1', 'local_x2', 'local_y2'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing columns in {}: {}'.format(detections_csv, ', '.join(sorted(missing))))

        for row in reader:
            rel_path = row['relative_patch_path']
            rows_by_path[rel_path].append(row)
    return rows_by_path


def draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: str) -> None:
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    bg = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(bg, fill=color)
    draw.text((x, y), text, fill='white', font=font)


def draw_boxes_on_tile(tile_path: str, detections: List[dict], out_path: str, color: str, width: int) -> None:
    image = Image.open(tile_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1 = float(det['local_x1'])
        y1 = float(det['local_y1'])
        x2 = float(det['local_x2'])
        y2 = float(det['local_y2'])
        score = float(det['score'])
        label = det.get('label_name') or det.get('label_id') or 'STAS'
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        draw_label(draw, x1, max(0, y1 - 24), '{} {:.3f}'.format(label, score), color)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Draw predicted STAS boxes on saved tile images.')
    parser.add_argument('--output_dir', default='/workspace/wsi_mask_tumor_predicted_tiles',
                        help='Inference output directory containing predicted_detections_global.csv and tile images.')
    parser.add_argument('--detections_csv', default=None,
                        help='Path to predicted_detections_global.csv. Defaults to output_dir/predicted_detections_global.csv')
    parser.add_argument('--tiles_root', default=None,
                        help='Root directory for relative_patch_path. Defaults to output_dir.')
    parser.add_argument('--boxed_dir', default=None,
                        help='Directory to write boxed images. Defaults to output_dir/boxed_tiles.')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Only draw detections at or above this score.')
    parser.add_argument('--color', default='red')
    parser.add_argument('--width', type=int, default=4)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    detections_csv = args.detections_csv or os.path.join(args.output_dir, 'predicted_detections_global.csv')
    tiles_root = args.tiles_root or args.output_dir
    boxed_dir = args.boxed_dir or os.path.join(args.output_dir, 'boxed_tiles')

    detections_by_path = load_detections(detections_csv)
    written = 0
    skipped_missing = 0
    skipped_empty = 0

    for rel_path, detections in detections_by_path.items():
        kept = [det for det in detections if float(det['score']) >= args.score_threshold]
        if not kept:
            skipped_empty += 1
            continue

        tile_path = os.path.join(tiles_root, rel_path)
        if not os.path.exists(tile_path):
            skipped_missing += 1
            continue

        out_path = os.path.join(boxed_dir, rel_path)
        draw_boxes_on_tile(tile_path, kept, out_path, args.color, args.width)
        written += 1

    print('[done] wrote boxed tiles: {}'.format(written))
    print('[done] skipped missing tiles: {}'.format(skipped_missing))
    print('[done] skipped below threshold: {}'.format(skipped_empty))
    print('[out] boxed_dir={}'.format(boxed_dir))


if __name__ == '__main__':
    main()