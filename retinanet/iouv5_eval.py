"""Shared helpers for RetinaNet prediction/evaluation with IoU_v5."""

import csv
import inspect
import json
import os

import numpy as np
import torch
from PIL import Image

import IoU_v5
from retinanet.input_preprocessing import resolve_imagenet_norm_mode

_MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
_STD = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def list_images(images_dir):
    """Return supported image file names in deterministic order."""
    return sorted(
        name for name in os.listdir(images_dir)
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    )


def load_gt_from_csv(csv_path):
    """Load CSV annotations into IoU_v5 GT-json format keyed by basename."""
    gt = {}
    with open(csv_path, 'r', newline='') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            path, x1, y1, x2, y2, _class_name = row[:6]
            name = os.path.basename(path)
            gt.setdefault(name, [])
            if (x1, y1, x2, y2, _class_name) == ('', '', '', '', ''):
                continue
            gt[name].append([int(x1), int(y1), int(x2), int(y2)])
    return gt


def build_gt_json(images_dir, csv_gt):
    """Build IoU_v5 GT json and include images with no annotations."""
    gt_raw = load_gt_from_csv(csv_gt)
    image_names = list_images(images_dir)
    return {name: gt_raw.get(name, []) for name in image_names}, image_names


def preprocess_image(img_rgb_u8, min_side=704, max_side=1920, imagenet_norm_mode='dataloader'):
    """Preprocess an RGB uint8 image using the dataloader Resizer convention."""
    import skimage.transform

    img = img_rgb_u8.astype(np.float32) / 255.0
    if imagenet_norm_mode == 'dataloader':
        img = (img - _MEAN) / _STD

    rows, cols, _channels = img.shape
    scale = min_side / min(rows, cols)
    if max(rows, cols) * scale > max_side:
        scale = max_side / max(rows, cols)

    new_h = int(round(rows * scale))
    new_w = int(round(cols * scale))
    img = skimage.transform.resize(img, (new_h, new_w))

    pad_h = 32 - (new_h % 32)
    pad_w = 32 - (new_w % 32)
    out = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.float32)
    out[:new_h, :new_w, :] = img
    return out, scale


def freeze_amp_norm_for_eval(net):
    """Freeze HarmoFL AmpNorm running amplitude if the checkpoint has it."""
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is not None:
        amp_norm.fix_amp = True
        print(f'[info] amp_norm_fix_amp=True running_amp_shape={tuple(amp_norm.running_amp.shape)}')


def disable_amp_norm_for_eval(net):
    """Disable HarmoFL AmpNorm for evaluation if the checkpoint has it."""
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is not None:
        input_preprocessor.amp_norm = None
        input_preprocessor.amp_norm_enabled = False
        print('[info] amp_norm disabled for eval/predict')


def load_model_for_inference(model_path, device=None, disable_amp_norm=False):
    """Load a checkpoint and return model, resolved device, and norm mode."""
    load_kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False
    net = torch.load(model_path, **load_kwargs)
    return prepare_model_for_inference(net, device=device, disable_amp_norm=disable_amp_norm)


def prepare_model_for_inference(net, device=None, disable_amp_norm=False):
    """Prepare an existing model object for IoU_v5 prediction."""
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


def preload_images(images_dir, image_names, min_side, max_side, imagenet_norm_mode):
    """Load and preprocess images once for deterministic prediction."""
    preloaded = []
    for index, name in enumerate(image_names, 1):
        img = np.array(Image.open(os.path.join(images_dir, name)).convert('RGB'))
        x, scale = preprocess_image(
            img,
            min_side=min_side,
            max_side=max_side,
            imagenet_norm_mode=imagenet_norm_mode,
        )
        preloaded.append({
            'name': name,
            'tensor': torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float(),
            'scale': scale,
        })
        if index % 20 == 0 or index == len(image_names):
            print(f'  [preload {index}/{len(image_names)}]')
    return preloaded


def _sort_and_limit_detections(dets, max_detections=None):
    dets = sorted(dets, key=lambda det: det[4], reverse=True)
    if max_detections is not None:
        dets = dets[:max_detections]
    return dets


def predict_json_for_model(net, preloaded_images, class_label=0, device=None,
                           progress_prefix='', max_detections=150):
    """Predict IoU_v5 JSON for a prepared model and preprocessed images."""
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
            pred_json[sample['name']] = _sort_and_limit_detections(dets, max_detections=max_detections)

            if index % 20 == 0 or index == len(preloaded_images):
                print(f'  [{progress_prefix}{index}/{len(preloaded_images)}] dets={len(pred_json[sample["name"]])}')

    return pred_json


def evaluate_prediction_json(gt_json, pred_json, conf_score, iou_threshold, check_inputs=True):
    """Evaluate IoU_v5 prediction json and return a metrics dictionary."""
    ap, best_box, pr_curve, max_idx = IoU_v5.get_precision_recall(
        gt_json,
        pred_json,
        classes=1,
        conf_score=conf_score,
        iou_threshold=iou_threshold,
        check_inputs=check_inputs,
    )
    precision, recall = pr_curve
    return {
        'ap': ap,
        'best_box': best_box,
        'precision': precision,
        'recall': recall,
        'best_index': max_idx,
        'best_precision': precision[max_idx] if precision else 0.0,
        'best_recall': recall[max_idx] if recall else 0.0,
    }


def write_json(path, data):
    """Write JSON data if path is provided."""
    if path:
        with open(path, 'w') as handle:
            json.dump(data, handle)


def evaluate_model_iouv5(net, images_dir, csv_gt, min_side=704, max_side=1920,
                         conf_score=0.05, iou_threshold=0.5, device=None,
                         out_pred_json=None, out_gt_json=None,
                         disable_amp_norm=False, class_label=0,
                         max_detections=150, check_inputs=True):
    """Evaluate an in-memory model with IoU_v5."""
    gt_json, image_names = build_gt_json(images_dir, csv_gt)
    net, device, imagenet_norm_mode = prepare_model_for_inference(
        net,
        device=device,
        disable_amp_norm=disable_amp_norm,
    )
    print(f'[info] imagenet_norm_mode={imagenet_norm_mode}')
    preloaded_images = preload_images(images_dir, image_names, min_side, max_side, imagenet_norm_mode)
    pred_json = predict_json_for_model(
        net,
        preloaded_images,
        class_label=class_label,
        device=device,
        max_detections=max_detections,
    )

    write_json(out_gt_json, gt_json)
    write_json(out_pred_json, pred_json)

    metrics = evaluate_prediction_json(
        gt_json,
        pred_json,
        conf_score=conf_score,
        iou_threshold=iou_threshold,
        check_inputs=check_inputs,
    )
    metrics['num_images'] = len(image_names)
    return metrics, gt_json, pred_json


def evaluate_checkpoint(model_path, images_dir, csv_gt, min_side=704, max_side=1920,
                        conf_score=0.05, iou_threshold=0.5, device=None,
                        out_pred_json=None, out_gt_json=None,
                        disable_amp_norm=False, class_label=0,
                        max_detections=150, check_inputs=True):
    """Load and evaluate a checkpoint with IoU_v5."""
    net, device, imagenet_norm_mode = load_model_for_inference(
        model_path,
        device=device,
        disable_amp_norm=disable_amp_norm,
    )
    gt_json, image_names = build_gt_json(images_dir, csv_gt)
    print(f'[info] imagenet_norm_mode={imagenet_norm_mode}')
    preloaded_images = preload_images(images_dir, image_names, min_side, max_side, imagenet_norm_mode)
    pred_json = predict_json_for_model(
        net,
        preloaded_images,
        class_label=class_label,
        device=device,
        max_detections=max_detections,
    )

    write_json(out_gt_json, gt_json)
    write_json(out_pred_json, pred_json)

    metrics = evaluate_prediction_json(
        gt_json,
        pred_json,
        conf_score=conf_score,
        iou_threshold=iou_threshold,
        check_inputs=check_inputs,
    )
    metrics['num_images'] = len(image_names)
    return metrics, gt_json, pred_json
