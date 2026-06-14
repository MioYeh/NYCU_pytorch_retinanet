#!/usr/bin/env python3
"""Full WSI RetinaNet inference after masking QuPath/PathCore tumor polygons."""

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import tifffile
import torch
import zarr

from retinanet.input_preprocessing import resolve_imagenet_norm_mode
from wsi_infer_on_the_fly import (
    build_tissue_mask,
    choose_level_for_target_mpp,
    disable_amp_norm_for_eval,
    extract_tile,
    find_wsi_files,
    freeze_amp_norm_for_eval,
    get_level_infos,
    load_classes,
    preprocess_for_retinanet,
    resolve_input_path,
    tile_tissue_ratio,
)
from wsi_infer_qupath_roi import parse_session_xml, point_in_any_roi


def find_matching_xml(wsi_path: str, xml_dir: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(wsi_path))[0]
    candidates = [
        os.path.join(xml_dir, stem + '.session.xml'),
        os.path.join(xml_dir, stem + '.xml'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def polygon_intersects_tile(
    points_level: Sequence[Tuple[float, float]],
    tile_w: int,
    tile_h: int,
) -> bool:
    xs = [point[0] for point in points_level]
    ys = [point[1] for point in points_level]
    return max(xs) >= 0 and max(ys) >= 0 and min(xs) <= tile_w and min(ys) <= tile_h


def mask_tumor_regions(
    tile_rgb: np.ndarray,
    rois,
    tile_x_level: int,
    tile_y_level: int,
    downsample_x: float,
    downsample_y: float,
    fill_rgb: Tuple[int, int, int],
) -> Tuple[np.ndarray, float]:
    image = Image.fromarray(tile_rgb.copy())
    draw = ImageDraw.Draw(image)
    tile_w, tile_h = image.size
    masked_polygons = 0

    for roi in rois:
        points_level = [
            ((base_x / downsample_x) - tile_x_level, (base_y / downsample_y) - tile_y_level)
            for base_x, base_y in roi.points_base
        ]
        if not polygon_intersects_tile(points_level, tile_w, tile_h):
            continue
        draw.polygon(points_level, fill=fill_rgb)
        masked_polygons += 1

    if masked_polygons == 0:
        return tile_rgb, 0.0

    masked_tile = np.asarray(image, dtype=np.uint8)
    changed = np.any(masked_tile != tile_rgb, axis=2)
    return masked_tile, float(changed.mean())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Full WSI inference with tumor polygons masked before prediction.')
    p.add_argument('--input_dir', default='/workspace/wsi_with_qupath')
    p.add_argument('--xml_dir', default='/workspace/wsi_with_qupath/sedeen',
                   help='Directory containing matching .session.xml files. Defaults to /workspace/wsi_with_qupath/sedeen.')
    p.add_argument('--output_dir', default='/workspace/wsi_mask_tumor_predicted_tiles')
    p.add_argument('--model_path', default='/workspace/csv_retinanet_29.pt')
    p.add_argument('--class_list', default='/workspace/STAS_VGH/classes.csv')

    p.add_argument('--target_mpp', type=float, default=0.5)
    p.add_argument('--tile_w', type=int, default=1920)
    p.add_argument('--tile_h', type=int, default=828)
    p.add_argument('--stride_w', type=int, default=960)
    p.add_argument('--stride_h', type=int, default=414)

    p.add_argument('--min_side', type=int, default=704)
    p.add_argument('--max_side', type=int, default=1920)
    p.add_argument('--score_threshold', type=float, default=0.05)
    p.add_argument('--target_label', type=int, default=0)
    p.add_argument('--disable_amp_norm', action='store_true')

    p.add_argument('--mask_fill', choices=['white', 'black', 'mean'], default='white',
                   help='Color used to cover tumor polygons before inference.')
    p.add_argument('--drop_detections_inside_tumor', action='store_true', default=True,
                   help='Drop detections whose center is still inside a tumor polygon.')
    p.add_argument('--keep_detections_inside_tumor', action='store_false', dest='drop_detections_inside_tumor')
    p.add_argument('--save_masked_tiles', action='store_true',
                   help='Save masked versions of positive tiles instead of the original unmasked tile.')

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


def resolve_fill_rgb(mask_fill: str) -> Tuple[int, int, int]:
    if mask_fill == 'white':
        return (255, 255, 255)
    if mask_fill == 'black':
        return (0, 0, 0)
    return (242, 242, 242)


def main() -> None:
    args = build_argparser().parse_args()
    args.xml_dir = args.xml_dir or os.path.join(args.input_dir, 'sedeen')
    os.makedirs(args.output_dir, exist_ok=True)

    args.model_path = resolve_input_path(args.model_path, known_candidates=['/workspace/csv_retinanet_29.pt'])
    args.class_list = resolve_input_path(args.class_list, known_candidates=['/workspace/STAS_VGH/classes.csv'])

    class_labels: Dict[int, str] = load_classes(args.class_list)
    fill_rgb = resolve_fill_rgb(args.mask_fill)

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
    if args.max_slides > 0:
        wsi_files = wsi_files[:args.max_slides]
    if not wsi_files:
        raise RuntimeError('No WSI TIFF found in {}'.format(args.input_dir))

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

            xml_path = find_matching_xml(slide_path, args.xml_dir)
            if xml_path is None:
                raise FileNotFoundError('No matching XML found for {}'.format(slide_name))
            rois, xml_dimensions = parse_session_xml(xml_path)
            if not rois:
                raise RuntimeError('No polygon ROI in {}'.format(xml_path))

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

                xs = list(range(0, w, args.stride_w))
                ys = list(range(0, h, args.stride_h))
                grid_total = len(xs) * len(ys)

                slide_seen = 0
                slide_tissue = 0
                slide_masked_tiles = 0
                slide_saved = 0
                slide_predicted = False
                slide_max_score = 0.0

                print('[slide {}/{}] {} rois={} level={} mpp=({:.3f},{:.3f}) size={}x{} grid_tiles={} mask_fill={}'.format(
                    slide_idx,
                    len(wsi_files),
                    slide_name,
                    len(rois),
                    picked_level.level,
                    picked_level.mpp_x,
                    picked_level.mpp_y,
                    w,
                    h,
                    grid_total,
                    args.mask_fill,
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

                        tile = extract_tile(level_array=level_array, x=x, y=y, tile_w=args.tile_w, tile_h=args.tile_h)
                        masked_tile, masked_fraction = mask_tumor_regions(
                            tile_rgb=tile,
                            rois=rois,
                            tile_x_level=x,
                            tile_y_level=y,
                            downsample_x=downsample_x,
                            downsample_y=downsample_y,
                            fill_rgb=fill_rgb,
                        )
                        if masked_fraction > 0:
                            slide_masked_tiles += 1

                        inp, scale = preprocess_for_retinanet(
                            tile_rgb=masked_tile,
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
                        if args.drop_detections_inside_tumor:
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
                        patch_name = '{}_L{}_x{}_y{}_tw{}_th{}_mask{:.3f}_s{:.3f}.{}'.format(
                            slide_stem,
                            picked_level.level,
                            x,
                            y,
                            args.tile_w,
                            args.tile_h,
                            masked_fraction,
                            max_score,
                            ext,
                        )
                        patch_path = os.path.join(slide_out_dir, patch_name)
                        rel_patch_path = os.path.relpath(patch_path, args.output_dir)

                        if not args.no_save_tiles:
                            save_tile = masked_tile if args.save_masked_tiles else tile
                            if args.save_format == 'jpg':
                                Image.fromarray(save_tile).save(patch_path, quality=args.jpg_quality)
                            else:
                                Image.fromarray(save_tile).save(patch_path)

                        slide_saved += 1
                        total_saved_tiles += 1
                        slide_predicted = True
                        slide_max_score = max(slide_max_score, max_score)

                        manifest_rows.append([
                            slide_name,
                            os.path.basename(xml_path),
                            str(picked_level.level),
                            str(round(picked_level.mpp_x, 6)),
                            str(round(picked_level.mpp_y, 6)),
                            str(x),
                            str(y),
                            str(args.tile_w),
                            str(args.tile_h),
                            str(round(x * downsample_x, 3)),
                            str(round(y * downsample_y, 3)),
                            str(round(ratio, 6)),
                            str(round(masked_fraction, 6)),
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
                            stop_slide = True
                            break

                        if slide_tissue % 200 == 0:
                            print('  progress tissue_tiles={} masked_tiles={} saved_tiles={}'.format(
                                slide_tissue,
                                slide_masked_tiles,
                                slide_saved,
                            ))

                    if stop_slide:
                        break

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
                    str(slide_seen),
                    str(slide_tissue),
                    str(slide_masked_tiles),
                    str(slide_saved),
                    '1' if slide_predicted else '0',
                    str(round(slide_max_score, 6)),
                ])

                print('  done seen={} tissue={} masked_tiles={} positive_tiles={} slide_has_stas={} max_score={:.6f}'.format(
                    slide_seen,
                    slide_tissue,
                    slide_masked_tiles,
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
            'level',
            'mpp_x',
            'mpp_y',
            'tile_x_level',
            'tile_y_level',
            'tile_w',
            'tile_h',
            'tile_x_base',
            'tile_y_base',
            'tissue_ratio',
            'masked_fraction',
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
            'tiles_seen',
            'tissue_tiles',
            'masked_tiles',
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