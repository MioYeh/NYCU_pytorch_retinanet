"""Evaluate a RetinaNet checkpoint on STAS_VGH using IoU_v5."""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/root/rebuild_retinanet/pytorch-retinanet')
sys.path.insert(0, '/root/rebuild_retinanet')

import IoU_v5  # noqa: E402
from retinanet.input_preprocessing import resolve_imagenet_norm_mode  # noqa: E402


_MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
_STD = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)


def preprocess(img_rgb_u8, min_side=608, max_side=1024, imagenet_norm_mode='dataloader'):
    import skimage.transform

    img = img_rgb_u8.astype(np.float32) / 255.0
    if imagenet_norm_mode == 'dataloader':
        img = (img - _MEAN) / _STD

    rows, cols, _ = img.shape
    scale = min_side / min(rows, cols)
    if max(rows, cols) * scale > max_side:
        scale = max_side / max(rows, cols)

    new_h = int(round(rows * scale))
    new_w = int(round(cols * scale))
    img = skimage.transform.resize(img, (new_h, new_w))

    # Match retinanet.dataloader.Resizer exactly. It pads by 32 even when the
    # resized side is already divisible by 32, so 828x1920 with 608/1024 becomes
    # 448x1056 instead of 448x1024.
    pad_h = 32 - (new_h % 32)
    pad_w = 32 - (new_w % 32)
    out = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.float32)
    out[:new_h, :new_w, :] = img
    return out, scale


def freeze_amp_norm_for_eval(net):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is not None:
        amp_norm.fix_amp = True
        print(f'[info] amp_norm_fix_amp=True running_amp_shape={tuple(amp_norm.running_amp.shape)}')


def disable_amp_norm_for_eval(net):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is not None:
        input_preprocessor.amp_norm = None
        input_preprocessor.amp_norm_enabled = False
        print('[info] amp_norm disabled for eval/predict')


def load_gt(csv_path):
    gt = {}
    with open(csv_path, 'r') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            path, x1, y1, x2, y2, _cls = row
            name = os.path.basename(path)
            gt.setdefault(name, []).append([int(x1), int(y1), int(x2), int(y2)])
    return gt


def list_images(images_dir):
    return sorted(
        name for name in os.listdir(images_dir)
        if name.lower().endswith(('.jpg', '.jpeg', '.png'))
    )


def build_gt_json(images_dir, csv_gt):
    gt_raw = load_gt(csv_gt)
    image_names = list_images(images_dir)
    return {name: gt_raw.get(name, []) for name in image_names}, image_names


def preload_images(images_dir, image_names, min_side, max_side, imagenet_norm_mode):
    preloaded = []
    for index, name in enumerate(image_names, 1):
        img = np.array(Image.open(os.path.join(images_dir, name)).convert('RGB'))
        x, scale = preprocess(img, min_side, max_side, imagenet_norm_mode=imagenet_norm_mode)
        preloaded.append({
            'name': name,
            'tensor': torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float(),
            'scale': scale,
        })
        if index % 20 == 0 or index == len(image_names):
            print(f'  [preload {index}/{len(image_names)}]')
    return preloaded


def load_model_for_inference(model_path, device=None, disable_amp_norm=False):
    net = torch.load(model_path, map_location='cpu', weights_only=False)
    net.eval()
    if disable_amp_norm:
        disable_amp_norm_for_eval(net)
    else:
        freeze_amp_norm_for_eval(net)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and torch.cuda.is_available():
        net = net.cuda()
    else:
        device = 'cpu'

    return net, device, resolve_imagenet_norm_mode(net)


def predict_json_for_model(net, preloaded_images, class_label=0, device=None, progress_prefix=''):
    pred_json = {}
    with torch.no_grad():
        for index, sample in enumerate(preloaded_images, 1):
            scores, labels, boxes = net(sample['tensor'].to(device))
            scale = sample['scale']

            if boxes.numel() > 0:
                boxes = boxes / scale
                mask = (labels == class_label)
                scores = scores[mask].cpu().numpy()
                boxes = boxes[mask].cpu().numpy()
            else:
                scores = np.zeros((0,), dtype=np.float32)
                boxes = np.zeros((0, 4), dtype=np.float32)

            dets = []
            for (x1, y1, x2, y2), score in zip(boxes, scores):
                dets.append([
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                    float(round(float(score), 6)),
                ])
            pred_json[sample['name']] = dets

            if index % 20 == 0 or index == len(preloaded_images):
                print(f'  [{progress_prefix}{index}/{len(preloaded_images)}] dets={len(dets)}')

    return pred_json


def evaluate_prediction_json(gt_json, pred_json, conf_score, iou_threshold):
    ap, best_box, pr_curve, max_idx = IoU_v5.get_precision_recall(
        gt_json,
        pred_json,
        classes=1,
        conf_score=conf_score,
        iou_threshold=iou_threshold,
    )
    precision, recall = pr_curve
    return {
        'ap': ap,
        'best_box': best_box,
        'precision': precision,
        'recall': recall,
        'best_index': max_idx,
        'best_precision': precision[max_idx],
        'best_recall': recall[max_idx],
    }


def evaluate_checkpoint(model_path, images_dir, csv_gt, min_side, max_side,
                        conf_score, iou_threshold, device=None,
                        out_pred_json=None, out_gt_json=None,
                        disable_amp_norm=False):
    gt_json, image_names = build_gt_json(images_dir, csv_gt)
    net, device, imagenet_norm_mode = load_model_for_inference(
        model_path,
        device=device,
        disable_amp_norm=disable_amp_norm,
    )
    print(f'[info] imagenet_norm_mode={imagenet_norm_mode}')
    preloaded_images = preload_images(images_dir, image_names, min_side, max_side, imagenet_norm_mode)
    pred_json = predict_json_for_model(net, preloaded_images, device=device)

    if out_gt_json:
        with open(out_gt_json, 'w') as handle:
            json.dump(gt_json, handle)
    if out_pred_json:
        with open(out_pred_json, 'w') as handle:
            json.dump(pred_json, handle)

    metrics = evaluate_prediction_json(gt_json, pred_json, conf_score, iou_threshold)
    metrics['num_images'] = len(image_names)
    return metrics, gt_json, pred_json


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
    parser.add_argument('--disable_amp_norm', action='store_true',
                        help='Disable HarmoFL AmpNorm during eval/predict, even if the checkpoint contains it.')
    parser.add_argument('--out_pred_json', default='/root/pytorch-retinanet/stas_vgh_pred.json')
    parser.add_argument('--out_gt_json', default='/root/pytorch-retinanet/stas_vgh_gt.json')
    args = parser.parse_args()

    print(f'[info] torch={torch.__version__}  cuda={torch.cuda.is_available()}')
    print(f'[info] model : {args.model_path}')
    print(f'[info] resize: min_side={args.min_side}  max_side={args.max_side}')

    metrics, gt_json, _pred_json = evaluate_checkpoint(
        args.model_path,
        args.images_dir,
        args.csv_gt,
        args.min_side,
        args.max_side,
        args.conf_score,
        args.iou_threshold,
        device=args.device,
        out_pred_json=args.out_pred_json,
        out_gt_json=args.out_gt_json,
        disable_amp_norm=args.disable_amp_norm,
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
