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
from utils import save_checkpoint, read_bad_samples

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
        image_f = image.flip(3)

        result = None
        for net in models:
            y_pred = net(image).cpu().detach().numpy()
            y_pred_f = net(image_f).flip(2).cpu().detach().numpy()

            y_pred = (y_pred + y_pred_f) * 0.5
            if result is None:
                result = y_pred
            else:
                result += y_pred
        result /= len(models)

        val_predictions.append(result)
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
        image_f = image.flip(3)

        result = None
        for net in models:
            y_pred = net(image).cpu().detach().numpy()
            y_pred_f = net(image_f).flip(2).cpu().detach().numpy()
            y_pred = (y_pred + y_pred_f) * 0.5
            if result is None:
                result = y_pred
            else:
                result += y_pred
        all_predictions.append(result / len(models))
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
    # use_list = [0, 1, 4, 8] -> 822
    # use_list = [4, 8] -> 819
    # use_list = [0, 1, 2, 4, 8] -> 823
    # use_list = [3, 5, 6, 7, 9] -> 825
    use_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    bad_samples = read_bad_samples()

    models = []

    data_path = os.sep.join(os.getcwd().split(os.sep)[:-1])

    validation_samples = {}
    for list_n in use_list:

        train_list_name = 'list_train' + str(list_n) + '_3600'
        validation_list_name = 'list_valid' + str(list_n) + '_400'
        _, samples = get_train_and_validation_samples_from_list(data_path, train_list_name,
                                                                                          validation_list_name)
        for key in samples.keys():
            validation_samples[key] = samples[key]

    list(map(validation_samples.__delitem__, filter(validation_samples.__contains__, bad_samples)))

    test_samples = get_test_samples(data_path)

    validation_ds = TGSSaltDataset(validation_samples, phase='validation')
    validation_dl = DataLoader(validation_ds, batch_size=16)

    test_ds = TGSSaltDatasetTest(test_samples)
    test_dl = DataLoader(test_ds, batch_size=16)

    for model_n in use_list:
        model_name = 'checkpoint'+str(model_n)+'.pth.tar'

        model_path = osp.join(data_path, model_name)

        checkpoint = torch.load(model_path)

        net = UNetResNet34().cuda()
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()

        models.append(net)

    score, threshold = eval_competition_score()
    print ('Score: ', score)
    test_submit(threshold)


