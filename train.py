import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import collections
import csv

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model, model_RESNET
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer, Mutil_Scale
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from retinanet import iouv5_eval

# print(torch.__version__.split('.')[0])
# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152, 200, 269', type=int, default=200)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('--output', help='Output folder to save weights and training log', required=True)
    parser.add_argument('--eval_metric', choices=['csv_eval', 'iouv5'], default='csv_eval',
                        help='Validation metric backend for CSV validation data.')
    parser.add_argument('--csv_val_images', help='Path to validation images directory when --eval_metric iouv5')
    parser.add_argument('--eval_min_side', type=int, default=704,
                        help='IoU_v5 eval resize smallest side.')
    parser.add_argument('--eval_max_side', type=int, default=1920,
                        help='IoU_v5 eval resize largest side cap.')
    parser.add_argument('--eval_conf_score', type=float, default=0.05,
                        help='IoU_v5 confidence threshold.')
    parser.add_argument('--eval_iou_threshold', type=float, default=0.5,
                        help='IoU_v5 IoU threshold.')
    parser.add_argument('--eval_class_label', type=int, default=0,
                        help='Model class label to export for IoU_v5 single-class evaluation.')
    parser.add_argument('--eval_max_detections', type=int, default=150,
                        help='Maximum detections per image for IoU_v5 JSON output.')
    parser.add_argument('--eval_disable_amp_norm', action='store_true',
                        help='Disable HarmoFL AmpNorm during IoU_v5 eval/predict.')
    parser.add_argument('--eval_skip_input_check', action='store_true',
                        help='Skip IoU_v5 prediction/GT JSON validation before metric calculation.')

    parser = parser.parse_args(args)

    # Create output directory (fail if already exists)
    if os.path.exists(parser.output):
        raise ValueError('Output folder "{}" already exists. Please use a different folder name.'.format(parser.output))
    os.makedirs(parser.output)
    print('Output folder created: {}'.format(parser.output))

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on CSV dataset.')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on CSV dataset.')

        # dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
        #                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Augmenter(), Normalizer(), Mutil_Scale()]))
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 200:
        retinanet = model_RESNET.resnest200(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 269:
        retinanet = model_RESNET.resnest269(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 0:
        done_model_path = '/workspace/weight_for_VGH_STAS/csv_retinanet_98.pt'
        retinanet = torch.load(done_model_path, weights_only=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152, 200, 269')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    log_csv_path = os.path.join(parser.output, 'training_log.csv')
    csv_file = open(log_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'cls_loss', 'reg_loss', 'total_loss', 'eval_metric', 'eval_AP'])
    csv_file.flush()

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        epoch_cls_loss = []
        epoch_reg_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                epoch_cls_loss.append(float(classification_loss))
                epoch_reg_loss.append(float(regression_loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

            mean_mAP = ''

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            if parser.eval_metric == 'iouv5':
                if parser.csv_val_images is None:
                    raise ValueError('--csv_val_images is required when --eval_metric iouv5')
                metrics, _gt_json, _pred_json = iouv5_eval.evaluate_model_iouv5(
                    retinanet,
                    parser.csv_val_images,
                    parser.csv_val,
                    min_side=parser.eval_min_side,
                    max_side=parser.eval_max_side,
                    conf_score=parser.eval_conf_score,
                    iou_threshold=parser.eval_iou_threshold,
                    class_label=parser.eval_class_label,
                    max_detections=parser.eval_max_detections,
                    disable_amp_norm=parser.eval_disable_amp_norm,
                    check_inputs=not parser.eval_skip_input_check,
                )
                mean_mAP = float(metrics['ap'])
                print(
                    '[result] IoU_v5 AP={:.5f} best-P={:.5f} best-R={:.5f}'.format(
                        metrics['ap'],
                        metrics['best_precision'],
                        metrics['best_recall'],
                    )
                )
            else:
                mAP = csv_eval.evaluate(dataset_val, retinanet)

                ap_values = [v[0] for v in mAP.values()]
                mean_mAP = float(np.mean(ap_values)) if ap_values else 0.0

        else:
            mean_mAP = ''

        scheduler.step(np.mean(epoch_loss))

        weight_path = os.path.join(parser.output, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
        torch.save(retinanet.module, weight_path)

        csv_writer.writerow([
            epoch_num,
            '{:.5f}'.format(np.mean(epoch_cls_loss)) if epoch_cls_loss else '',
            '{:.5f}'.format(np.mean(epoch_reg_loss)) if epoch_reg_loss else '',
            '{:.5f}'.format(np.mean(epoch_loss)) if epoch_loss else '',
            parser.eval_metric if mean_mAP != '' else '',
            '{:.5f}'.format(mean_mAP) if mean_mAP != '' else '',
        ])
        csv_file.flush()

    csv_file.close()
    print('Training log saved to: {}'.format(log_csv_path))

    retinanet.eval()

    final_path = os.path.join(parser.output, 'model_final.pt')
    torch.save(retinanet, final_path)
    print('Final model saved to: {}'.format(final_path))


if __name__ == '__main__':
    main()
