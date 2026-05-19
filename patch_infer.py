#!/usr/bin/env python3
"""RetinaNet inference on pre-cut patch images.

Input directory layout:
    input_dir/
        <slide_name>/
            <x1>_<y1>_<x2>_<y2>.png   (global WSI coordinates encoded in filename)
            ...

Outputs (written to output_dir):
    predicted_tiles_manifest.csv   -- one row per positive patch
    predicted_detections_global.csv -- one row per detection box (global coords)
    slide_summary.csv              -- one row per slide
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retinanet.input_preprocessing import resolve_imagenet_norm_mode  # noqa: E402

MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
STD  = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (same as wsi_infer_on_the_fly / eval_stas_vgh_iouv5)
# ---------------------------------------------------------------------------

def preprocess_for_retinanet(tile_rgb, min_side, max_side, imagenet_norm_mode='dataloader'):
    rows, cols, _ = tile_rgb.shape
    scale = min_side / float(min(rows, cols))
    if max(rows, cols) * scale > max_side:
        scale = max_side / float(max(rows, cols))

    new_h = int(round(rows * scale))
    new_w = int(round(cols * scale))
    resized = np.asarray(
        Image.fromarray(tile_rgb).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.float32,
    )
    resized = resized / 255.0
    if imagenet_norm_mode == 'dataloader':
        resized = (resized - MEAN) / STD

    pad_h = (32 - (new_h % 32)) % 32
    pad_w = (32 - (new_w % 32)) % 32
    canvas = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.float32)
    canvas[:new_h, :new_w, :] = resized

    tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
    return tensor, scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r'^(\d+)_(\d+)_(\d+)_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)


def parse_coords_from_name(filename):
    """Return (x1, y1, x2, y2) from '<x1>_<y1>_<x2>_<y2>.png', or None."""
    m = _COORD_RE.match(os.path.basename(filename))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_classes(path):
    labels = {}
    if not path or not os.path.exists(path):
        return labels
    with open(path, 'r', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            cls_name, cls_id = row[0], int(row[1])
            labels[cls_id] = cls_name
    return labels


def collect_slides(input_dir):
    """Return list of (slide_name, [patch_path, ...]) sorted by slide_name."""
    slides = []
    for name in sorted(os.listdir(input_dir)):
        slide_dir = os.path.join(input_dir, name)
        if not os.path.isdir(slide_dir):
            continue
        patches = sorted(
            os.path.join(slide_dir, f)
            for f in os.listdir(slide_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        if patches:
            slides.append((name, patches))
    return slides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description='RetinaNet inference on pre-cut patches.')
    p.add_argument('--input_dir',  default='/workspace/with',
                   help='Root dir; each subfolder = one slide.')
    p.add_argument('--output_dir', default='/workspace/with_predictions',
                   help='Where to write CSV summaries.')
    p.add_argument('--model_path', required=True,
                   help='Path to saved RetinaNet .pt checkpoint.')
    p.add_argument('--class_list', default='/workspace/STAS_VGH/classes.csv',
                   help='CSV of class_name,class_id (optional).')
    p.add_argument('--min_side',   type=int,   default=704)
    p.add_argument('--max_side',   type=int,   default=1920)
    p.add_argument('--score_threshold', type=float, default=0.3)
    p.add_argument('--target_label',    type=int,   default=0,
                   help='Only keep this class label. -1 = all labels.')
    p.add_argument('--max_slides', type=int, default=0,
                   help='Process only first N slides. 0 = all.')
    p.add_argument('--device', default=None,
                   help='cuda / cpu. Auto-detects if omitted.')
    p.add_argument('--disable_amp_norm', action='store_true')
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[info] device={device}')
    print(f'[info] loading model: {args.model_path}')

    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()

    # amp_norm handling (same as wsi_infer_on_the_fly)
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None) if input_preprocessor else None
    if amp_norm is not None:
        if args.disable_amp_norm:
            input_preprocessor.amp_norm = None
            input_preprocessor.amp_norm_enabled = False
            print('[info] amp_norm disabled')
        else:
            amp_norm.fix_amp = True
            print(f'[info] amp_norm fix_amp=True shape={tuple(amp_norm.running_amp.shape)}')

    imagenet_norm_mode = resolve_imagenet_norm_mode(model)
    print(f'[info] imagenet_norm_mode={imagenet_norm_mode}')

    class_labels = load_classes(args.class_list)
    slides = collect_slides(args.input_dir)
    if args.max_slides > 0:
        slides = slides[:args.max_slides]
    print(f'[info] slides={len(slides)}  score_threshold={args.score_threshold}')

    manifest_rows = []
    det_rows      = []
    slide_rows    = []
    total_positive_slides = 0

    with torch.no_grad():
        for slide_idx, (slide_name, patches) in enumerate(slides, 1):
            print(f'[slide {slide_idx}/{len(slides)}] {slide_name}  patches={len(patches)}')

            slide_positive = False
            slide_max_score = 0.0
            slide_saved = 0

            for patch_path in patches:
                coords = parse_coords_from_name(patch_path)
                # coords may be None if filename doesn't match pattern
                tile_x1 = coords[0] if coords else 0
                tile_y1 = coords[1] if coords else 0

                tile_rgb = np.array(Image.open(patch_path).convert('RGB'))
                tile_h, tile_w = tile_rgb.shape[:2]

                inp, scale = preprocess_for_retinanet(
                    tile_rgb, args.min_side, args.max_side, imagenet_norm_mode
                )
                inp = inp.to(device=device, dtype=torch.float32)

                scores, labels, boxes = model(inp)
                if scores.numel() == 0:
                    continue

                scores_np = scores.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                boxes_np  = boxes.detach().cpu().numpy() / scale

                keep = scores_np >= args.score_threshold
                if args.target_label >= 0:
                    keep &= (labels_np == args.target_label)
                if not np.any(keep):
                    continue

                kept_scores = scores_np[keep]
                kept_labels = labels_np[keep]
                kept_boxes  = boxes_np[keep]
                max_score   = float(np.max(kept_scores))

                rel_patch = os.path.relpath(patch_path, args.input_dir)

                manifest_rows.append([
                    slide_name,
                    str(tile_x1),
                    str(tile_y1),
                    str(tile_w),
                    str(tile_h),
                    str(len(kept_scores)),
                    str(round(max_score, 6)),
                    rel_patch,
                ])

                for score, label, box in zip(kept_scores, kept_labels, kept_boxes):
                    lx1, ly1, lx2, ly2 = box.tolist()
                    gx1 = tile_x1 + lx1
                    gy1 = tile_y1 + ly1
                    gx2 = tile_x1 + lx2
                    gy2 = tile_y1 + ly2
                    cls_name = class_labels.get(int(label), str(int(label)))
                    det_rows.append([
                        rel_patch,
                        slide_name,
                        str(int(label)),
                        cls_name,
                        str(round(float(score), 6)),
                        str(round(float(lx1), 3)),
                        str(round(float(ly1), 3)),
                        str(round(float(lx2), 3)),
                        str(round(float(ly2), 3)),
                        str(round(float(gx1), 3)),
                        str(round(float(gy1), 3)),
                        str(round(float(gx2), 3)),
                        str(round(float(gy2), 3)),
                    ])

                slide_positive = True
                slide_max_score = max(slide_max_score, max_score)
                slide_saved += 1

            if slide_positive:
                total_positive_slides += 1

            slide_rows.append([
                slide_name,
                str(len(patches)),
                str(slide_saved),
                '1' if slide_positive else '0',
                str(round(slide_max_score, 6)),
            ])
            print(f'  patches={len(patches)}  positive_tiles={slide_saved}  '
                  f'slide_has_stas={int(slide_positive)}  max_score={slide_max_score:.4f}')

    # ---- Write CSVs ----
    manifest_path = os.path.join(args.output_dir, 'predicted_tiles_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slide_name', 'tile_x1', 'tile_y1', 'tile_w', 'tile_h',
                    'num_detections', 'max_score', 'relative_patch_path'])
        w.writerows(manifest_rows)

    det_path = os.path.join(args.output_dir, 'predicted_detections_global.csv')
    with open(det_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['relative_patch_path', 'slide_name', 'label_id', 'label_name',
                    'score', 'local_x1', 'local_y1', 'local_x2', 'local_y2',
                    'global_x1', 'global_y1', 'global_x2', 'global_y2'])
        w.writerows(det_rows)

    slide_summary_path = os.path.join(args.output_dir, 'slide_summary.csv')
    with open(slide_summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slide_name', 'total_patches', 'positive_tiles',
                    'slide_has_stas', 'max_score'])
        w.writerows(slide_rows)

    print(f'\n[done] slides={len(slides)}  positive_slides={total_positive_slides}/{len(slides)}')
    print(f'[out] {manifest_path}')
    print(f'[out] {det_path}')
    print(f'[out] {slide_summary_path}')


if __name__ == '__main__':
    main()
