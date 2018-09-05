import os
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
from data import TGSSaltDataset, get_train_and_validation_datasets
from utils import save_checkpoint

from tensorboardX import SummaryWriter

if __name__ == '__main__':

    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    net = UNetResNet34().cuda()
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    data_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    training_samples, validation_samples = get_train_and_validation_datasets(data_path)

    train_ds = TGSSaltDataset(training_samples,phase='train')
    validation_ds = TGSSaltDataset(validation_samples,phase='validation')

    train_dl = DataLoader(train_ds,batch_size=16,shuffle=True)
    validation_dl = DataLoader(validation_ds,batch_size=8,shuffle=False)

    train_writer = SummaryWriter('logs/train')
    validation_writer = SummaryWriter('logs/val')

    step = 0
    best_validation_loss = 1e9
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
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
            loss = criterion(outputs, labels)
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

        validation_batches = 0
        running_loss = 0.0
        net.eval()
        for i, data in enumerate(validation_dl):
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

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
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet34',
            'state_dict': net.state_dict(),
            'best_prec1': running_loss,
            'optimizer': optimizer.state_dict(),
        }, data_path, False)

        if best_validation_loss > running_loss:
            best_validation_loss = running_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet34',
                'state_dict': net.state_dict(),
                'best_prec1': running_loss,
                'optimizer': optimizer.state_dict(),
            }, data_path, True)
    print('Finished Training')

