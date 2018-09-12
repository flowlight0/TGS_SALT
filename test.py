import os
import os.path as osp
import random
import numpy as np

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

from iou_loss import lovasz_hinge
from models import UNetResNet34
from data import TGSSaltDataset, TGSSaltDatasetTest
from data import get_train_and_validation_samples, get_test_samples
from data import get_train_and_validation_samples_from_list
from utils import save_checkpoint

from tensorboardX import SummaryWriter

from augmentation import compute_center_pad

import pandas as pd

def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()

def eval_competition_score():
    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101

    val_predictions = []
    val_masks = []
    for image, mask in validation_dl:
        image = image.cuda()
        y_pred = net(image).cpu().detach().numpy()

        image_f = image.flip(3)
        y_pred_f = net(image_f).flip(2).cpu().detach().numpy()

        y_pred = (y_pred + y_pred_f) * 0.5

        val_predictions.append(y_pred)
        val_masks.append(mask.detach().numpy())

    val_predictions_stacked = np.vstack(val_predictions)

    val_masks_stacked = np.vstack(val_masks)
    val_predictions_stacked = val_predictions_stacked[:, Y0:Y1, X0:X1]
    val_masks_stacked = val_masks_stacked[:, Y0:Y1, X0:X1]

    assert val_masks_stacked.shape == val_predictions_stacked.shape

    val_masks_stacked = (val_masks_stacked > 0.5).astype(int)

    metric_by_threshold = []
    for threshold in np.linspace(0.4, 0.5, 11):
        val_binary_prediction = (val_predictions_stacked > threshold).astype(int)
        accuracies = iou_numpy(val_binary_prediction, val_masks_stacked)
        print('Threshold: %.2f, Metric: %.5f' % (threshold, np.mean(accuracies)))
        metric_by_threshold.append((np.mean(accuracies), threshold))

    best_metric, best_threshold = max(metric_by_threshold)
    print ('Eval competition score: ', best_metric, best_threshold)
    return best_metric, best_threshold

def test_submit(threshold):
    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101

    all_predictions = []
    for image in test_dl:
        image = image.cuda()
        y_pred = net(image).cpu().detach().numpy()
        image_f = image.flip(3)
        y_pred_f = net(image_f).flip(2).cpu().detach().numpy()
        y_pred = (y_pred + y_pred_f) * 0.5
        all_predictions.append(y_pred)
    all_predictions_stacked = np.vstack(all_predictions)
    all_predictions_stacked = all_predictions_stacked[:, Y0:Y1, X0:X1]

    binary_prediction = (all_predictions_stacked > threshold).astype(int)
    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    all_masks = []
    for p_mask in list(binary_prediction):
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))

    submit = pd.DataFrame([test_samples.keys(), all_masks]).T
    submit.columns = ['id', 'rle_mask']
    submit.to_csv('submit.csv', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--use_list', type=int, default=-1)
    parser.add_argument('--load_weights', type=str, default='')
    args = parser.parse_args()

    data_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    if args.use_list == -1:
        training_samples, validation_samples = get_train_and_validation_samples(data_path)
    else:
        train_list_name = 'list_train' + str(args.use_list) + '_3600'
        validation_list_name = 'list_valid' + str(args.use_list) + '_400'
        training_samples, validation_samples = get_train_and_validation_samples_from_list(data_path, train_list_name,
                                                                                          validation_list_name)
    test_samples = get_test_samples(data_path)

    validation_ds = TGSSaltDataset(validation_samples, phase='validation')
    validation_dl = DataLoader(validation_ds, batch_size=16)

    test_ds = TGSSaltDatasetTest(test_samples)
    test_dl = DataLoader(test_ds, batch_size=16)

    model_path = osp.join(data_path, args.load_weights)

    checkpoint = torch.load(model_path)

    net = UNetResNet34().cuda()
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    score, threshold = eval_competition_score()
    print ('Score: ', score)
    test_submit(threshold)


