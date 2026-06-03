import argparse
import torch
from torchvision import transforms

from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval, iouv5_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple validation script for a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path', help='Path to images directory', type=str)
    parser.add_argument('--class_list_path', help='Path to classlist csv', type=str)
    parser.add_argument('--iou_threshold', help='IOU threshold used for evaluation', type=float, default=0.5)
    parser.add_argument('--metric', choices=['csv_eval', 'iouv5'], default='csv_eval',
                        help='Metric backend. Use iouv5 to produce JSON predictions and calculate with IoU_v5.py.')
    parser.add_argument('--score_threshold', type=float, default=0.05,
                        help='Score threshold for csv_eval and IoU_v5 conf_score.')
    parser.add_argument('--max_detections', type=int, default=150,
                        help='Maximum detections per image for IoU_v5 JSON output.')
    parser.add_argument('--min_side', type=int, default=704,
                        help='Resize smallest side for IoU_v5 prediction.')
    parser.add_argument('--max_side', type=int, default=1920,
                        help='Resize largest side cap for IoU_v5 prediction.')
    parser.add_argument('--class_label', type=int, default=0,
                        help='Model class label to export for IoU_v5 single-class evaluation.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--disable_amp_norm', action='store_true',
                        help='Disable HarmoFL AmpNorm during IoU_v5 eval/predict, even if the checkpoint contains it.')
    parser.add_argument('--skip_input_check', action='store_true',
                        help='Skip IoU_v5 prediction/GT JSON validation before metric calculation.')
    parser.add_argument('--out_pred_json', default=None,
                        help='Optional path to write IoU_v5 prediction JSON.')
    parser.add_argument('--out_gt_json', default=None,
                        help='Optional path to write IoU_v5 GT JSON.')
    parser = parser.parse_args(args)

    if parser.metric == 'iouv5':
        if parser.images_path is None:
            raise ValueError('--images_path is required when --metric iouv5')
        metrics, _gt_json, _pred_json = iouv5_eval.evaluate_checkpoint(
            parser.model_path,
            parser.images_path,
            parser.csv_annotations_path,
            min_side=parser.min_side,
            max_side=parser.max_side,
            conf_score=parser.score_threshold,
            iou_threshold=parser.iou_threshold,
            device=parser.device,
            out_pred_json=parser.out_pred_json,
            out_gt_json=parser.out_gt_json,
            disable_amp_norm=parser.disable_amp_norm,
            class_label=parser.class_label,
            max_detections=parser.max_detections,
            check_inputs=not parser.skip_input_check,
        )
        print(
            f"[result] IoU_v5 AP={metrics['ap']:.4f}  "
            f"best-P={metrics['best_precision']:.4f}  "
            f"best-R={metrics['best_recall']:.4f}  "
            f"num_images={metrics['num_images']}"
        )
        return

    dataset_val = CSVDataset(
        parser.csv_annotations_path,
        parser.class_list_path,
        transform=transforms.Compose([Normalizer(), Resizer()]),
    )
    retinanet = torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    print(csv_eval.evaluate(
        dataset_val,
        retinanet,
        iou_threshold=parser.iou_threshold,
        score_threshold=parser.score_threshold,
    ))


if __name__ == '__main__':
    main()
