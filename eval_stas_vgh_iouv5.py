"""Evaluate a RetinaNet checkpoint on STAS_VGH using IoU_v5."""

import argparse

import torch

from retinanet import iouv5_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default='/workspace/STAS_VGH/Test_Images')
    parser.add_argument('--csv_gt', default='/workspace/STAS_VGH/test_annotations_fixed.csv')
    parser.add_argument('--model_path', default='/workspace/csv_retinanet_29.pt')
    parser.add_argument('--min_side', type=int, default=608)
    parser.add_argument('--max_side', type=int, default=1024)
    parser.add_argument('--conf_score', type=float, default=0.05)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--device', default=None)
    parser.add_argument('--class_label', type=int, default=0)
    parser.add_argument('--max_detections', type=int, default=150)
    parser.add_argument('--disable_amp_norm', action='store_true',
                        help='Disable HarmoFL AmpNorm during eval/predict, even if the checkpoint contains it.')
    parser.add_argument('--skip_input_check', action='store_true',
                        help='Skip IoU_v5 prediction/GT JSON validation before metric calculation.')
    parser.add_argument('--out_pred_json', default='/root/pytorch-retinanet/stas_vgh_pred.json')
    parser.add_argument('--out_gt_json', default='/root/pytorch-retinanet/stas_vgh_gt.json')
    args = parser.parse_args()

    print(f'[info] torch={torch.__version__}  cuda={torch.cuda.is_available()}')
    print(f'[info] model : {args.model_path}')
    print(f'[info] resize: min_side={args.min_side}  max_side={args.max_side}')
    print(f'[info] class_label={args.class_label}  max_detections={args.max_detections}')

    metrics, gt_json, _pred_json = iouv5_eval.evaluate_checkpoint(
        args.model_path,
        args.images_dir,
        args.csv_gt,
        min_side=args.min_side,
        max_side=args.max_side,
        conf_score=args.conf_score,
        iou_threshold=args.iou_threshold,
        device=args.device,
        out_pred_json=args.out_pred_json,
        out_gt_json=args.out_gt_json,
        disable_amp_norm=args.disable_amp_norm,
        class_label=args.class_label,
        max_detections=args.max_detections,
        check_inputs=not args.skip_input_check,
    )

    print(f'[info] wrote {args.out_gt_json}, {args.out_pred_json}')
    print(f'[info] num images (dir): {metrics["num_images"]}  (with gt: {len(gt_json)})')
    print(
        f"[result] AP={metrics['ap']:.4f}  "
        f"(best-F1 index={metrics['best_index']}, "
        f"best-P={metrics['best_precision']:.4f}, "
        f"best-R={metrics['best_recall']:.4f})"
    )


if __name__ == '__main__':
    main()
