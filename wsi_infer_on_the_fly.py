#!/usr/bin/env python3
"""On-the-fly WSI tiling + RetinaNet inference.

Workflow:
1) Read WSI tile directly from TIFF pyramid (no pre-cut dataset on disk).
2) Filter with lightweight tissue mask.
3) Run RetinaNet inference tile-by-tile.
4) Save only tiles that have predictions above threshold.

Saved tile filenames include slide/level/x/y so they can be traced back for
future fine-tuning.
"""

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tifffile
from retinanet.input_preprocessing import resolve_imagenet_norm_mode
import torch
import zarr


MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
STD = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)


@dataclass
class LevelInfo:
    level: int
    width: int
    height: int
    mpp_x: float
    mpp_y: float


def parse_first_pixel_spacing_mm(image_description: str) -> Tuple[float, float]:
    pattern = r'DICOM_PIXEL_SPACING".*?&quot;([0-9.]+)&quot;\s*&quot;([0-9.]+)&quot;'
    m = re.search(pattern, image_description)
    if not m:
        raise ValueError('Cannot find DICOM_PIXEL_SPACING in ImageDescription')
    return float(m.group(1)), float(m.group(2))


def get_level_infos(tif: tifffile.TiffFile) -> List[LevelInfo]:
    series = tif.series[0]
    if not series.levels:
        raise ValueError('This TIFF has no pyramid levels')

    desc_tag = tif.pages[0].tags.get('ImageDescription')
    if desc_tag is None:
        raise ValueError('Missing ImageDescription metadata')

    spacing_mm_y, spacing_mm_x = parse_first_pixel_spacing_mm(desc_tag.value)
    base_mpp_y = spacing_mm_y * 1000.0
    base_mpp_x = spacing_mm_x * 1000.0

    base_h, base_w, _ = series.levels[0].shape
    out = []
    for idx, lvl in enumerate(series.levels):
        h, w, _ = lvl.shape
        ds_x = base_w / float(w)
        ds_y = base_h / float(h)
        out.append(
            LevelInfo(
                level=idx,
                width=w,
                height=h,
                mpp_x=base_mpp_x * ds_x,
                mpp_y=base_mpp_y * ds_y,
            )
        )
    return out


def choose_level_for_target_mpp(levels: Sequence[LevelInfo], target_mpp: float) -> LevelInfo:
    return min(levels, key=lambda x: abs(((x.mpp_x + x.mpp_y) * 0.5) - target_mpp))


def build_tissue_mask(
    level_array: zarr.Array,
    width: int,
    height: int,
    max_mask_side: int,
    sat_threshold: float,
    intensity_threshold: float,
) -> Tuple[np.ndarray, float, float]:
    thumb_scale = max(width, height) / float(max_mask_side)
    thumb_scale = max(thumb_scale, 1.0)
    stride = max(1, int(math.ceil(thumb_scale)))

    thumb = np.asarray(level_array[0:height:stride, 0:width:stride, :], dtype=np.uint8)
    rgb = thumb.astype(np.float32) / 255.0

    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    sat = (max_c - min_c) / (max_c + 1e-6)
    gray = rgb.mean(axis=2)

    tissue_mask = (sat >= sat_threshold) & (gray <= intensity_threshold)

    r_minus_g = rgb[:, :, 0] - rgb[:, :, 1]
    b_minus_g = rgb[:, :, 2] - rgb[:, :, 1]
    tissue_mask |= ((r_minus_g > 0.015) & (b_minus_g > 0.005) & (gray < 0.95))

    mask_h, mask_w = tissue_mask.shape
    scale_x = width / float(mask_w)
    scale_y = height / float(mask_h)
    return tissue_mask, scale_x, scale_y


def tile_tissue_ratio(
    tissue_mask: np.ndarray,
    scale_x: float,
    scale_y: float,
    x: int,
    y: int,
    tile_w: int,
    tile_h: int,
) -> float:
    mask_h, mask_w = tissue_mask.shape
    x1 = max(0, min(mask_w, int(math.floor(x / scale_x))))
    y1 = max(0, min(mask_h, int(math.floor(y / scale_y))))
    x2 = max(0, min(mask_w, int(math.ceil((x + tile_w) / scale_x))))
    y2 = max(0, min(mask_h, int(math.ceil((y + tile_h) / scale_y))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = tissue_mask[y1:y2, x1:x2]
    return float(roi.mean())


def extract_tile(level_array: zarr.Array, x: int, y: int, tile_w: int, tile_h: int) -> np.ndarray:
    h, w, _ = level_array.shape
    x2 = min(x + tile_w, w)
    y2 = min(y + tile_h, h)
    crop = np.asarray(level_array[y:y2, x:x2, :], dtype=np.uint8)
    if crop.shape[0] == tile_h and crop.shape[1] == tile_w:
        return crop

    out = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    out[: crop.shape[0], : crop.shape[1], :] = crop
    return out


def preprocess_for_retinanet(
    tile_rgb: np.ndarray,
    min_side: int,
    max_side: int,
    imagenet_norm_mode: str = 'dataloader',
) -> Tuple[torch.Tensor, float]:
    """Match retinanet Normalizer + Resizer behavior."""
    rows, cols, _ = tile_rgb.shape
    smallest_side = min(rows, cols)
    largest_side = max(rows, cols)

    scale = min_side / float(smallest_side)
    if largest_side * scale > max_side:
        scale = max_side / float(largest_side)

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


def find_wsi_files(input_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(('.tif', '.tiff')):
            files.append(os.path.join(input_dir, name))
    return files


def load_classes(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    labels: Dict[int, str] = {}
    with open(path, 'r', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            cls_name, cls_id = row[0], int(row[1])
            labels[cls_id] = cls_name
    return labels


def freeze_amp_norm_for_eval(model: torch.nn.Module) -> None:
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is not None:
        amp_norm.fix_amp = True
        print('[info] amp_norm_fix_amp=True running_amp_shape={}'.format(tuple(amp_norm.running_amp.shape)))


def disable_amp_norm_for_eval(model: torch.nn.Module) -> None:
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is not None:
        input_preprocessor.amp_norm = None
        input_preprocessor.amp_norm_enabled = False
        print('[info] amp_norm disabled for WSI inference')


def resolve_input_path(path: str, known_candidates: Optional[Sequence[str]] = None) -> str:
    """Resolve CLI path with fallbacks for common workspace locations."""
    if os.path.exists(path):
        return path

    tried = []
    candidates: List[str] = []

    if not os.path.isabs(path):
        candidates.append(os.path.abspath(path))
        candidates.append(os.path.join(os.getcwd(), path))
        candidates.append(os.path.join(os.path.dirname(__file__), path))

    if known_candidates:
        candidates.extend(known_candidates)

    seen = set()
    uniq_candidates = []
    for c in candidates:
        c_abs = os.path.abspath(c)
        if c_abs in seen:
            continue
        seen.add(c_abs)
        uniq_candidates.append(c_abs)

    for cand in uniq_candidates:
        tried.append(cand)
        if os.path.exists(cand):
            print('[info] path not found: {} -> using {}'.format(path, cand))
            return cand

    msg = 'File not found: {}\nTried:\n  - {}'.format(path, '\n  - '.join(tried) if tried else '(no candidates)')
    raise FileNotFoundError(msg)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='On-the-fly WSI inference and save only predicted tiles.')
    p.add_argument('--input_dir', default='/workspace/no_stas_wsi')
    p.add_argument('--output_dir', default='/workspace/no_stas_wsi_predicted_tiles')
    p.add_argument('--model_path', default='/workspace/csv_retinanet_29.pt')
    p.add_argument('--class_list', default='/workspace/STAS_VGH/classes.csv')

    p.add_argument('--target_mpp', type=float, default=0.5)
    p.add_argument('--tile_w', type=int, default=1920)
    p.add_argument('--tile_h', type=int, default=828)
    p.add_argument('--stride_w', type=int, default=1920)
    p.add_argument('--stride_h', type=int, default=828)

    p.add_argument('--min_side', type=int, default=704)
    p.add_argument('--max_side', type=int, default=1920)

    p.add_argument('--score_threshold', type=float, default=0.3)
    p.add_argument('--target_label', type=int, default=0,
                   help='Keep only this class label. Set -1 for any class.')
    p.add_argument('--disable_amp_norm', action='store_true',
                   help='Disable HarmoFL AmpNorm during WSI inference, even if the checkpoint contains it.')

    p.add_argument('--min_tissue_ratio', type=float, default=0.03)
    p.add_argument('--tissue_mask_side', type=int, default=2048)
    p.add_argument('--sat_threshold', type=float, default=0.035)
    p.add_argument('--intensity_threshold', type=float, default=0.93)

    p.add_argument('--max_slides', type=int, default=0,
                   help='0 means all slides.')
    p.add_argument('--max_tissue_tiles_per_slide', type=int, default=0,
                   help='0 means no limit.')
    p.add_argument('--no_save_tiles', action='store_true',
                   help='Do not save predicted tile images; still write CSV summaries.')
    p.add_argument('--save_format', choices=['png', 'jpg'], default='png')
    p.add_argument('--jpg_quality', type=int, default=95)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.model_path = resolve_input_path(
        args.model_path,
        known_candidates=['/workspace/csv_retinanet_29.pt'],
    )
    args.class_list = resolve_input_path(
        args.class_list,
        known_candidates=['/workspace/STAS_VGH/classes.csv'],
    )

    class_labels = load_classes(args.class_list)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[info] device=', device)
    print('[info] loading model:', args.model_path)
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()
    model.training = False
    if args.disable_amp_norm:
        disable_amp_norm_for_eval(model)
    else:
        freeze_amp_norm_for_eval(model)
    imagenet_norm_mode = resolve_imagenet_norm_mode(model)
    print('[info] imagenet_norm_mode=', imagenet_norm_mode)

    wsi_files = find_wsi_files(args.input_dir)
    if not wsi_files:
        raise RuntimeError('No WSI TIFF found in {}'.format(args.input_dir))
    if args.max_slides > 0:
        wsi_files = wsi_files[:args.max_slides]

    print('[info] slides=', len(wsi_files))
    print('[info] tile={}x{} stride={}x{} target_mpp={}'.format(
        args.tile_w, args.tile_h, args.stride_w, args.stride_h, args.target_mpp
    ))
    print('[info] score_threshold={} target_label={}'.format(args.score_threshold, args.target_label))

    manifest_rows: List[List[str]] = []
    det_rows: List[List[str]] = []
    slide_rows: List[List[str]] = []

    total_tiles_seen = 0
    total_tissue_tiles = 0
    total_saved_tiles = 0
    total_positive_slides = 0

    with torch.no_grad():
        for slide_idx, slide_path in enumerate(wsi_files, 1):
            slide_name = os.path.basename(slide_path)
            slide_stem = os.path.splitext(slide_name)[0]
            slide_out_dir = os.path.join(args.output_dir, slide_stem)
            os.makedirs(slide_out_dir, exist_ok=True)

            with tifffile.TiffFile(slide_path) as tif:
                level_infos = get_level_infos(tif)
                picked_level = choose_level_for_target_mpp(level_infos, args.target_mpp)

                level = tif.series[0].levels[picked_level.level]
                level_array = zarr.open(level.aszarr(), mode='r')
                h, w, _ = level_array.shape

                tissue_mask, scale_x, scale_y = build_tissue_mask(
                    level_array=level_array,
                    width=w,
                    height=h,
                    max_mask_side=args.tissue_mask_side,
                    sat_threshold=args.sat_threshold,
                    intensity_threshold=args.intensity_threshold,
                )

                xs = list(range(0, w, args.stride_w))
                ys = list(range(0, h, args.stride_h))
                grid_total = len(xs) * len(ys)

                slide_seen = 0
                slide_tissue = 0
                slide_saved = 0
                slide_predicted = False
                slide_max_score = 0.0

                print('[slide {}/{}] {} level={} mpp=({:.3f},{:.3f}) size={}x{} grid_tiles={}'.format(
                    slide_idx,
                    len(wsi_files),
                    slide_name,
                    picked_level.level,
                    picked_level.mpp_x,
                    picked_level.mpp_y,
                    w,
                    h,
                    grid_total,
                ))

                stop_slide = False
                for y in ys:
                    for x in xs:
                        slide_seen += 1
                        total_tiles_seen += 1

                        ratio = tile_tissue_ratio(
                            tissue_mask=tissue_mask,
                            scale_x=scale_x,
                            scale_y=scale_y,
                            x=x,
                            y=y,
                            tile_w=args.tile_w,
                            tile_h=args.tile_h,
                        )
                        if ratio < args.min_tissue_ratio:
                            continue

                        slide_tissue += 1
                        total_tissue_tiles += 1

                        tile = extract_tile(
                            level_array=level_array,
                            x=x,
                            y=y,
                            tile_w=args.tile_w,
                            tile_h=args.tile_h,
                        )
                        inp, scale = preprocess_for_retinanet(
                            tile_rgb=tile,
                            min_side=args.min_side,
                            max_side=args.max_side,
                            imagenet_norm_mode=imagenet_norm_mode,
                        )
                        inp = inp.to(device=device, dtype=torch.float32)

                        scores, labels, boxes = model(inp)
                        if scores.numel() == 0:
                            continue

                        scores = scores.detach().cpu().numpy()
                        labels = labels.detach().cpu().numpy()
                        boxes = boxes.detach().cpu().numpy() / scale

                        keep = scores >= args.score_threshold
                        if args.target_label >= 0:
                            keep &= (labels == args.target_label)

                        if not np.any(keep):
                            continue

                        kept_scores = scores[keep]
                        kept_labels = labels[keep]
                        kept_boxes = boxes[keep]
                        max_score = float(np.max(kept_scores))

                        ext = args.save_format
                        patch_name = (
                            '{}_L{}_x{}_y{}_tw{}_th{}_s{:.3f}.{}'.format(
                                slide_stem,
                                picked_level.level,
                                x,
                                y,
                                args.tile_w,
                                args.tile_h,
                                max_score,
                                ext,
                            )
                        )
                        patch_path = os.path.join(slide_out_dir, patch_name)
                        rel_patch_path = os.path.relpath(patch_path, args.output_dir)

                        if not args.no_save_tiles:
                            if args.save_format == 'jpg':
                                Image.fromarray(tile).save(patch_path, quality=args.jpg_quality)
                            else:
                                Image.fromarray(tile).save(patch_path)

                        slide_saved += 1
                        total_saved_tiles += 1
                        slide_predicted = True
                        slide_max_score = max(slide_max_score, max_score)

                        manifest_rows.append([
                            slide_name,
                            str(picked_level.level),
                            str(round(picked_level.mpp_x, 6)),
                            str(round(picked_level.mpp_y, 6)),
                            str(x),
                            str(y),
                            str(args.tile_w),
                            str(args.tile_h),
                            str(round(ratio, 6)),
                            str(len(kept_scores)),
                            str(round(max_score, 6)),
                            rel_patch_path,
                        ])

                        for score, label, box in zip(kept_scores, kept_labels, kept_boxes):
                            x1, y1, x2, y2 = box.tolist()
                            gx1 = x + x1
                            gy1 = y + y1
                            gx2 = x + x2
                            gy2 = y + y2
                            cls_name = class_labels.get(int(label), str(int(label)))
                            det_rows.append([
                                rel_patch_path,
                                slide_name,
                                str(int(label)),
                                cls_name,
                                str(round(float(score), 6)),
                                str(round(float(x1), 3)),
                                str(round(float(y1), 3)),
                                str(round(float(x2), 3)),
                                str(round(float(y2), 3)),
                                str(round(float(gx1), 3)),
                                str(round(float(gy1), 3)),
                                str(round(float(gx2), 3)),
                                str(round(float(gy2), 3)),
                            ])

                        if args.max_tissue_tiles_per_slide > 0 and slide_tissue >= args.max_tissue_tiles_per_slide:
                            stop_slide = True
                            break

                        if slide_tissue % 200 == 0:
                            print('  progress tissue_tiles={} saved_tiles={}'.format(slide_tissue, slide_saved))

                    if stop_slide:
                        break

                if slide_predicted:
                    total_positive_slides += 1

                slide_rows.append([
                    slide_name,
                    str(picked_level.level),
                    str(round(picked_level.mpp_x, 6)),
                    str(round(picked_level.mpp_y, 6)),
                    str(slide_seen),
                    str(slide_tissue),
                    str(slide_saved),
                    '1' if slide_predicted else '0',
                    str(round(slide_max_score, 6)),
                ])

                print('  done seen={} tissue={} positive_tiles={} slide_has_stas={} max_score={:.6f}'.format(
                    slide_seen,
                    slide_tissue,
                    slide_saved,
                    int(slide_predicted),
                    slide_max_score,
                ))

    manifest_path = os.path.join(args.output_dir, 'predicted_tiles_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow([
            'slide_name',
            'level',
            'mpp_x',
            'mpp_y',
            'tile_x',
            'tile_y',
            'tile_w',
            'tile_h',
            'tissue_ratio',
            'num_detections',
            'max_score',
            'relative_patch_path',
        ])
        wtr.writerows(manifest_rows)

    det_path = os.path.join(args.output_dir, 'predicted_detections_global.csv')
    with open(det_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow([
            'relative_patch_path',
            'slide_name',
            'label_id',
            'label_name',
            'score',
            'local_x1',
            'local_y1',
            'local_x2',
            'local_y2',
            'global_x1',
            'global_y1',
            'global_x2',
            'global_y2',
        ])
        wtr.writerows(det_rows)

    slide_summary_path = os.path.join(args.output_dir, 'slide_summary.csv')
    with open(slide_summary_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow([
            'slide_name',
            'level',
            'mpp_x',
            'mpp_y',
            'tiles_seen',
            'tissue_tiles',
            'positive_tiles',
            'slide_has_stas',
            'max_score',
        ])
        wtr.writerows(slide_rows)

    print('[done] slides={} tiles_seen={} tissue_tiles={} saved_tiles={}'.format(
        len(wsi_files),
        total_tiles_seen,
        total_tissue_tiles,
        total_saved_tiles,
    ))
    print('[done] positive_slides={}/{}'.format(total_positive_slides, len(wsi_files)))
    print('[out] manifest={}'.format(manifest_path))
    print('[out] detections={}'.format(det_path))
    print('[out] slide_summary={}'.format(slide_summary_path))


if __name__ == '__main__':
    main()