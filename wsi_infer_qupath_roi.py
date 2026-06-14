#!/usr/bin/env python3
"""Run RetinaNet WSI inference only around QuPath/PathCore polygon ROIs."""

import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tifffile
import torch
import zarr

from retinanet.input_preprocessing import resolve_imagenet_norm_mode
from wsi_infer_on_the_fly import (
    build_tissue_mask,
    choose_level_for_target_mpp,
    disable_amp_norm_for_eval,
    extract_tile,
    freeze_amp_norm_for_eval,
    get_level_infos,
    load_classes,
    preprocess_for_retinanet,
    resolve_input_path,
    tile_tissue_ratio,
)


@dataclass
class PolygonROI:
    name: str
    points_base: List[Tuple[float, float]]

    def bbox_base(self) -> Tuple[float, float, float, float]:
        xs = [point[0] for point in self.points_base]
        ys = [point[1] for point in self.points_base]
        return min(xs), min(ys), max(xs), max(ys)


def parse_session_xml(xml_path: str) -> Tuple[List[PolygonROI], Optional[Tuple[int, int]]]:
    root = ET.parse(xml_path).getroot()

    dimensions = None
    dimensions_node = root.find('.//dimensions')
    if dimensions_node is not None and dimensions_node.text:
        width_text, height_text = dimensions_node.text.strip().split(',')[:2]
        dimensions = (int(float(width_text)), int(float(height_text)))

    rois: List[PolygonROI] = []
    for index, graphic in enumerate(root.findall('.//graphic')):
        if graphic.attrib.get('type', '').lower() != 'polygon':
            continue

        points: List[Tuple[float, float]] = []
        for point_node in graphic.findall('.//point'):
            if not point_node.text:
                continue
            x_text, y_text = point_node.text.strip().split(',')[:2]
            points.append((float(x_text), float(y_text)))

        if len(points) >= 3:
            rois.append(PolygonROI(graphic.attrib.get('name', 'roi_{}'.format(index)), points))

    return rois, dimensions


def find_wsi_xml_pairs(input_dir: str) -> List[Tuple[str, str]]:
    files = sorted(os.listdir(input_dir))
    wsi_paths = [
        os.path.join(input_dir, name)
        for name in files
        if name.lower().endswith(('.tif', '.tiff'))
    ]
    xml_paths = [
        os.path.join(input_dir, name)
        for name in files
        if name.lower().endswith('.xml')
    ]
    xml_by_stem = {os.path.basename(path).replace('.session.xml', '').replace('.xml', ''): path for path in xml_paths}

    pairs: List[Tuple[str, str]] = []
    for wsi_path in wsi_paths:
        stem = os.path.splitext(os.path.basename(wsi_path))[0]
        xml_path = xml_by_stem.get(stem)
        if xml_path is None:
            session_name = stem + '.session.xml'
            fallback = os.path.join(input_dir, session_name)
            if os.path.exists(fallback):
                xml_path = fallback
        if xml_path is None:
            print('[warn] no XML found for {}, skipping'.format(os.path.basename(wsi_path)))
            continue
        pairs.append((wsi_path, xml_path))

    return pairs


def clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def point_in_polygon(x: float, y: float, points: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    j = len(points) - 1
    for i, point_i in enumerate(points):
        xi, yi = point_i
        xj, yj = points[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_any_roi(x: float, y: float, rois: Sequence[PolygonROI]) -> bool:
    return any(point_in_polygon(x, y, roi.points_base) for roi in rois)


def sampled_roi_fraction_base(
    rois: Sequence[PolygonROI],
    tile_x_level: int,
    tile_y_level: int,
    tile_w: int,
    tile_h: int,
    downsample_x: float,
    downsample_y: float,
    grid_size: int,
) -> float:
    grid_size = max(1, grid_size)
    inside = 0
    total = grid_size * grid_size
    for row in range(grid_size):
        sample_y_level = tile_y_level + ((row + 0.5) / grid_size) * tile_h
        sample_y_base = sample_y_level * downsample_y
        for col in range(grid_size):
            sample_x_level = tile_x_level + ((col + 0.5) / grid_size) * tile_w
            sample_x_base = sample_x_level * downsample_x
            if point_in_any_roi(sample_x_base, sample_y_base, rois):
                inside += 1
    return inside / float(total)


def build_roi_tile_origins(
    rois: Sequence[PolygonROI],
    level_width: int,
    level_height: int,
    downsample_x: float,
    downsample_y: float,
    tile_w: int,
    tile_h: int,
    stride_w: int,
    stride_h: int,
    roi_margin_base: int,
) -> List[Tuple[int, int, int]]:
    origins = set()
    max_x_origin = max(0, level_width - tile_w)
    max_y_origin = max(0, level_height - tile_h)

    for roi_index, roi in enumerate(rois):
        min_x, min_y, max_x, max_y = roi.bbox_base()
        min_x = max(0.0, min_x - roi_margin_base)
        min_y = max(0.0, min_y - roi_margin_base)
        max_x = min(level_width * downsample_x, max_x + roi_margin_base)
        max_y = min(level_height * downsample_y, max_y + roi_margin_base)

        level_x1 = clamp_int(math.floor(min_x / downsample_x), 0, max_x_origin)
        level_y1 = clamp_int(math.floor(min_y / downsample_y), 0, max_y_origin)
        level_x2 = clamp_int(math.ceil(max_x / downsample_x), 0, level_width)
        level_y2 = clamp_int(math.ceil(max_y / downsample_y), 0, level_height)

        start_x = max(0, (level_x1 // stride_w) * stride_w)
        start_y = max(0, (level_y1 // stride_h) * stride_h)
        end_x = min(max_x_origin, level_x2)
        end_y = min(max_y_origin, level_y2)

        x = start_x
        while x <= end_x:
            y = start_y
            while y <= end_y:
                origins.add((x, y, roi_index))
                y += stride_h
            x += stride_w

    return sorted(origins, key=lambda item: (item[2], item[1], item[0]))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='ROI-limited WSI inference using PathCore/QuPath XML polygons.')
    p.add_argument('--input_dir', default='/workspace/wsi_with_qupath')
    p.add_argument('--output_dir', default='/workspace/wsi_with_qupath_roi_predicted_tiles')
    p.add_argument('--model_path', default='/workspace/csv_retinanet_29.pt')
    p.add_argument('--class_list', default='/workspace/STAS_VGH/classes.csv')

    p.add_argument('--target_mpp', type=float, default=0.5)
    p.add_argument('--tile_w', type=int, default=1920)
    p.add_argument('--tile_h', type=int, default=828)
    p.add_argument('--stride_w', type=int, default=960)
    p.add_argument('--stride_h', type=int, default=414)
    p.add_argument('--roi_margin_base', type=int, default=4096,
                   help='Margin around tumor ROI in level-0/base pixels.')
    p.add_argument('--include_roi_core', action='store_true',
                   help='Also scan/count detections inside tumor polygons. By default tumor polygons are excluded.')
    p.add_argument('--max_roi_tile_fraction', type=float, default=0.50,
                   help='Skip a tile when sampled tumor-polygon coverage is above this fraction.')
    p.add_argument('--roi_filter_grid', type=int, default=7,
                   help='Sampling grid size used to estimate tumor-polygon coverage inside a tile.')

    p.add_argument('--min_side', type=int, default=704)
    p.add_argument('--max_side', type=int, default=1920)
    p.add_argument('--score_threshold', type=float, default=0.05)
    p.add_argument('--target_label', type=int, default=0)
    p.add_argument('--disable_amp_norm', action='store_true')

    p.add_argument('--min_tissue_ratio', type=float, default=0.03)
    p.add_argument('--tissue_mask_side', type=int, default=2048)
    p.add_argument('--sat_threshold', type=float, default=0.035)
    p.add_argument('--intensity_threshold', type=float, default=0.93)

    p.add_argument('--max_slides', type=int, default=0)
    p.add_argument('--max_tissue_tiles_per_slide', type=int, default=0)
    p.add_argument('--no_save_tiles', action='store_true')
    p.add_argument('--save_format', choices=['png', 'jpg'], default='png')
    p.add_argument('--jpg_quality', type=int, default=95)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.model_path = resolve_input_path(args.model_path, known_candidates=['/workspace/csv_retinanet_29.pt'])
    args.class_list = resolve_input_path(args.class_list, known_candidates=['/workspace/STAS_VGH/classes.csv'])

    class_labels: Dict[int, str] = load_classes(args.class_list)

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

    pairs = find_wsi_xml_pairs(args.input_dir)
    if args.max_slides > 0:
        pairs = pairs[:args.max_slides]
    if not pairs:
        raise RuntimeError('No WSI/XML pairs found in {}'.format(args.input_dir))

    manifest_rows: List[List[str]] = []
    det_rows: List[List[str]] = []
    slide_rows: List[List[str]] = []

    total_tiles_seen = 0
    total_tissue_tiles = 0
    total_saved_tiles = 0
    total_positive_slides = 0

    with torch.no_grad():
        for slide_idx, (slide_path, xml_path) in enumerate(pairs, 1):
            slide_name = os.path.basename(slide_path)
            slide_stem = os.path.splitext(slide_name)[0]
            slide_out_dir = os.path.join(args.output_dir, slide_stem)
            os.makedirs(slide_out_dir, exist_ok=True)

            rois, xml_dimensions = parse_session_xml(xml_path)
            if not rois:
                print('[warn] no polygon ROI in {}, skipping'.format(os.path.basename(xml_path)))
                continue

            with tifffile.TiffFile(slide_path) as tif:
                level_infos = get_level_infos(tif)
                picked_level = choose_level_for_target_mpp(level_infos, args.target_mpp)
                base_h, base_w, _ = tif.series[0].levels[0].shape
                if xml_dimensions and xml_dimensions != (base_w, base_h):
                    print('[warn] XML dimensions {} do not match level0 {}'.format(xml_dimensions, (base_w, base_h)))

                level = tif.series[0].levels[picked_level.level]
                level_array = zarr.open(level.aszarr(), mode='r')
                h, w, _ = level_array.shape
                downsample_x = base_w / float(w)
                downsample_y = base_h / float(h)

                tissue_mask, scale_x, scale_y = build_tissue_mask(
                    level_array=level_array,
                    width=w,
                    height=h,
                    max_mask_side=args.tissue_mask_side,
                    sat_threshold=args.sat_threshold,
                    intensity_threshold=args.intensity_threshold,
                )
                tile_origins = build_roi_tile_origins(
                    rois=rois,
                    level_width=w,
                    level_height=h,
                    downsample_x=downsample_x,
                    downsample_y=downsample_y,
                    tile_w=args.tile_w,
                    tile_h=args.tile_h,
                    stride_w=args.stride_w,
                    stride_h=args.stride_h,
                    roi_margin_base=args.roi_margin_base,
                )

                slide_seen = 0
                slide_tissue = 0
                slide_saved = 0
                slide_roi_skipped = 0
                slide_predicted = False
                slide_max_score = 0.0

                print('[slide {}/{}] {} rois={} level={} mpp=({:.3f},{:.3f}) size={}x{} roi_tiles={}'.format(
                    slide_idx,
                    len(pairs),
                    slide_name,
                    len(rois),
                    picked_level.level,
                    picked_level.mpp_x,
                    picked_level.mpp_y,
                    w,
                    h,
                    len(tile_origins),
                ))

                for x, y, roi_index in tile_origins:
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

                    roi_fraction = 0.0
                    if not args.include_roi_core:
                        roi_fraction = sampled_roi_fraction_base(
                            rois=rois,
                            tile_x_level=x,
                            tile_y_level=y,
                            tile_w=args.tile_w,
                            tile_h=args.tile_h,
                            downsample_x=downsample_x,
                            downsample_y=downsample_y,
                            grid_size=args.roi_filter_grid,
                        )
                        if roi_fraction > args.max_roi_tile_fraction:
                            slide_roi_skipped += 1
                            continue

                    slide_tissue += 1
                    total_tissue_tiles += 1

                    tile = extract_tile(level_array=level_array, x=x, y=y, tile_w=args.tile_w, tile_h=args.tile_h)
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

                    scores_np = scores.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    boxes_np = boxes.detach().cpu().numpy() / scale

                    keep = scores_np >= args.score_threshold
                    if args.target_label >= 0:
                        keep &= labels_np == args.target_label
                    if not args.include_roi_core:
                        outside_roi = []
                        for box in boxes_np:
                            center_x_level = x + ((box[0] + box[2]) * 0.5)
                            center_y_level = y + ((box[1] + box[3]) * 0.5)
                            outside_roi.append(
                                not point_in_any_roi(
                                    center_x_level * downsample_x,
                                    center_y_level * downsample_y,
                                    rois,
                                )
                            )
                        keep &= np.array(outside_roi, dtype=bool)
                    if not np.any(keep):
                        continue

                    kept_scores = scores_np[keep]
                    kept_labels = labels_np[keep]
                    kept_boxes = boxes_np[keep]
                    max_score = float(np.max(kept_scores))

                    ext = args.save_format
                    patch_name = '{}_roi{}_L{}_x{}_y{}_tw{}_th{}_s{:.3f}.{}'.format(
                        slide_stem,
                        roi_index,
                        picked_level.level,
                        x,
                        y,
                        args.tile_w,
                        args.tile_h,
                        max_score,
                        ext,
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
                        os.path.basename(xml_path),
                        str(roi_index),
                        rois[roi_index].name,
                        str(picked_level.level),
                        str(round(picked_level.mpp_x, 6)),
                        str(round(picked_level.mpp_y, 6)),
                        str(x),
                        str(y),
                        str(args.tile_w),
                        str(args.tile_h),
                        str(round(x * downsample_x, 3)),
                        str(round(y * downsample_y, 3)),
                        str(round(roi_fraction, 6)),
                        str(round(ratio, 6)),
                        str(len(kept_scores)),
                        str(round(max_score, 6)),
                        rel_patch_path,
                    ])

                    for score, label, box in zip(kept_scores, kept_labels, kept_boxes):
                        x1, y1, x2, y2 = box.tolist()
                        level_x1 = x + x1
                        level_y1 = y + y1
                        level_x2 = x + x2
                        level_y2 = y + y2
                        cls_name = class_labels.get(int(label), str(int(label)))
                        det_rows.append([
                            rel_patch_path,
                            slide_name,
                            str(roi_index),
                            str(int(label)),
                            cls_name,
                            str(round(float(score), 6)),
                            str(round(float(x1), 3)),
                            str(round(float(y1), 3)),
                            str(round(float(x2), 3)),
                            str(round(float(y2), 3)),
                            str(round(float(level_x1), 3)),
                            str(round(float(level_y1), 3)),
                            str(round(float(level_x2), 3)),
                            str(round(float(level_y2), 3)),
                            str(round(float(level_x1 * downsample_x), 3)),
                            str(round(float(level_y1 * downsample_y), 3)),
                            str(round(float(level_x2 * downsample_x), 3)),
                            str(round(float(level_y2 * downsample_y), 3)),
                        ])

                    if args.max_tissue_tiles_per_slide > 0 and slide_tissue >= args.max_tissue_tiles_per_slide:
                        print('  reached max_tissue_tiles_per_slide={}'.format(args.max_tissue_tiles_per_slide))
                        break

                    if slide_tissue % 200 == 0:
                        print('  progress tissue_tiles={} saved_tiles={}'.format(slide_tissue, slide_saved))

                if slide_predicted:
                    total_positive_slides += 1

                slide_rows.append([
                    slide_name,
                    os.path.basename(xml_path),
                    str(len(rois)),
                    str(picked_level.level),
                    str(round(picked_level.mpp_x, 6)),
                    str(round(picked_level.mpp_y, 6)),
                    str(round(downsample_x, 6)),
                    str(round(downsample_y, 6)),
                    str(len(tile_origins)),
                    str(slide_seen),
                    str(slide_roi_skipped),
                    str(slide_tissue),
                    str(slide_saved),
                    '1' if slide_predicted else '0',
                    str(round(slide_max_score, 6)),
                ])

                print('  done roi_tiles={} seen={} skipped_tumor_core={} tissue={} positive_tiles={} slide_has_stas={} max_score={:.6f}'.format(
                    len(tile_origins),
                    slide_seen,
                    slide_roi_skipped,
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
            'xml_name',
            'roi_index',
            'roi_name',
            'level',
            'mpp_x',
            'mpp_y',
            'tile_x_level',
            'tile_y_level',
            'tile_w',
            'tile_h',
            'tile_x_base',
            'tile_y_base',
            'tile_tumor_fraction',
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
            'roi_index',
            'label_id',
            'label_name',
            'score',
            'local_x1',
            'local_y1',
            'local_x2',
            'local_y2',
            'level_x1',
            'level_y1',
            'level_x2',
            'level_y2',
            'base_x1',
            'base_y1',
            'base_x2',
            'base_y2',
        ])
        wtr.writerows(det_rows)

    slide_summary_path = os.path.join(args.output_dir, 'slide_summary.csv')
    with open(slide_summary_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow([
            'slide_name',
            'xml_name',
            'num_rois',
            'level',
            'mpp_x',
            'mpp_y',
            'downsample_x',
            'downsample_y',
            'roi_candidate_tiles',
            'tiles_seen',
            'tumor_core_skipped_tiles',
            'tissue_tiles',
            'positive_tiles',
            'slide_has_stas',
            'max_score',
        ])
        wtr.writerows(slide_rows)

    print('[done] slides={} tiles_seen={} tissue_tiles={} saved_tiles={}'.format(
        len(pairs),
        total_tiles_seen,
        total_tissue_tiles,
        total_saved_tiles,
    ))
    print('[done] positive_slides={}/{}'.format(total_positive_slides, len(pairs)))
    print('[out] manifest={}'.format(manifest_path))
    print('[out] detections={}'.format(det_path))
    print('[out] slide_summary={}'.format(slide_summary_path))


if __name__ == '__main__':
    main()