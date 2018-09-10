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

from iou_loss import lovasz_hinge, FocalLoss2d
from models import UNetResNet34
from data import TGSSaltDataset, TGSSaltDatasetTest
from data import get_train_and_validation_samples, get_test_samples
from data import get_train_and_validation_samples_from_list
from utils import save_checkpoint

from tensorboardX import SummaryWriter

from augmentation import compute_center_pad

def eval_competition_score():
    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101

    val_predictions = []
    val_masks = []
    for image, mask in validation_dl:
        image = image.cuda()
        y_pred = net(image).cpu().detach().numpy()
        val_predictions.append(y_pred)
        val_masks.append(mask.detach().numpy())

    val_predictions_stacked = np.vstack(val_predictions)

    val_masks_stacked = np.vstack(val_masks)
    val_predictions_stacked = val_predictions_stacked[:, Y0:Y1, X0:X1]
    val_masks_stacked = val_masks_stacked[:, Y0:Y1, X0:X1]

    assert val_masks_stacked.shape == val_predictions_stacked.shape

    from sklearn.metrics import jaccard_similarity_score

    metric_by_threshold = []
    for threshold in np.linspace(0, 1, 11):
        val_binary_prediction = (val_predictions_stacked > threshold).astype(int)

        iou_values = []
        for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
            iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
            iou_values.append(iou)
        iou_values = np.array(iou_values)

        accuracies = [
            np.mean(iou_values > iou_threshold)
            for iou_threshold in np.linspace(0.5, 0.95, 10)
        ]
        print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
        metric_by_threshold.append((np.mean(accuracies), threshold))

    best_metric, best_threshold = max(metric_by_threshold)
    print ('Eval competition score: ', best_metric, best_threshold)
    return best_metric, best_threshold


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--use_list',  type=int, default=-1)
    parser.add_argument('--load_weights', type=str, default='')
    args = parser.parse_args()

    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101
    data_path = os.sep.join(os.getcwd().split(os.sep)[:-1])


    criterion = FocalLoss2d()
    net = UNetResNet34().cuda()
    if args.load_weights:
        model_path = osp.join(data_path, args.load_weights)
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['state_dict'])
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    if args.use_list == -1:
        training_samples, validation_samples = get_train_and_validation_samples(data_path)
    else:
        train_list_name = 'list_train'+str(args.use_list)+'_3600'
        validation_list_name = 'list_valid'+str(args.use_list)+'_400'
        training_samples, validation_samples = get_train_and_validation_samples_from_list(data_path, train_list_name, validation_list_name)

    train_ds = TGSSaltDataset(training_samples,phase='train')
    validation_ds = TGSSaltDataset(validation_samples,phase='validation')

    train_dl = DataLoader(train_ds,batch_size=16,shuffle=True)
    validation_dl = DataLoader(validation_ds,batch_size=8,shuffle=False)

    train_writer = SummaryWriter('logs/train')
    validation_writer = SummaryWriter('logs/val')

    step = 0
    best_competition_score = 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=55, gamma=0.1)
    for epoch in range(60):  # loop over the dataset multiple times
        scheduler.step()
        print ('Epoch: ', epoch)

        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_dl):
            step += 1
            # get the inputs
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs[:, Y0:Y1, X0:X1], labels[:,Y0:Y1, X0:X1])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_writer.add_scalar('Loss', loss.item(), step)
            if i % 20 == 19:
                outputs = F.sigmoid(outputs)
                comb = torch.cat([labels, outputs], dim=1)
                comb = vutils.make_grid(comb, scale_each=True)
                comb = comb.unsqueeze(1)
                train_writer.add_image('Image', comb, step)

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        with torch.no_grad():
            validation_batches = 0
            running_loss = 0.0
            net.eval()
            for i, data in enumerate(validation_dl):
                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs[:,Y0:Y1,X0:X1], labels[:,Y0:Y1,X0:X1])

                # print statistics
                running_loss += loss.item()
                validation_batches += 1

                if random.random() < 0.5:
                    outputs = F.sigmoid(outputs)
                    comb = torch.cat([labels, outputs], dim=1)
                    comb = vutils.make_grid(comb, scale_each=True)
                    comb = comb.unsqueeze(1)
                    validation_writer.add_image('Image', comb, step)

            running_loss /= validation_batches
            validation_writer.add_scalar('Loss', running_loss, step)

            comp_score, threshold = eval_competition_score()
            validation_writer.add_scalar('Competition score', comp_score, step)
            validation_writer.add_scalar('Threshold for competion score', threshold, step)

            if best_competition_score < comp_score:
                best_competition_score = comp_score
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'resnet34',
                    'state_dict': net.state_dict(),
                    'best_prec1': comp_score,
                    'optimizer': optimizer.state_dict(),
                }, data_path, True)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet34',
            'state_dict': net.state_dict(),
            'best_prec1': running_loss,
            'optimizer': optimizer.state_dict(),
        }, data_path, False)

    print('Finished Training')

