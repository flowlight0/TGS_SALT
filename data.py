import os
import sys
import glob
import random
import os.path as osp

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from utils import get_file_id, Sample
from utils import read_image, read_mask

def get_all_samples(root_path, has_target=True):
    samples = {}

    image_folder = osp.join(root_path, 'images')
    image_files = glob.glob(osp.join(image_folder,'*.png'))

    for image_file in image_files:
        sample_id = get_file_id(image_file)
        samples[sample_id] = Sample()
        samples[sample_id].image_fp = image_file

    if has_target:
        mask_folder = osp.join(root_path, 'masks')
        mask_files = glob.glob(osp.join(mask_folder,'*.png'))

        for mask_file in mask_files:
            sample_id = get_file_id(mask_file)
            assert sample_id in samples, 'get all samples, wrong mask id'

            samples[sample_id].mask_fp = mask_file

    return samples

def get_test_samples(root_path):
    data_path = osp.join(root_path, osp.join('data', 'test'))
    samples = get_all_samples(data_path, has_target=False)
    return samples

def read_from_list(file, data_path):
    ids = []
    with open(file, 'r') as phase_input:
        lines = phase_input.readlines()
        for line in lines:
            splt = line.split('/')[1]
            ids.append(splt)
    train_data = osp.join(data_path, 'train')
    image_folder = osp.join(train_data, 'images')
    mask_folder = osp.join(train_data, 'masks')

    samples = {}
    for id in ids:
        id = id[:-1]
        samples[id] = Sample()
        samples[id].image_fp = osp.join(image_folder, id + '.png')
        samples[id].mask_fp = osp.join(mask_folder, id + '.png')
    return samples

def get_train_and_validation_samples_from_list(root_path, train_file, validation_file):
    data_path = osp.join(root_path, 'data')
    train_file = osp.join(data_path, osp.join('split', train_file))
    validation_file = osp.join(data_path, osp.join('split', validation_file))

    train_samples = read_from_list(train_file, data_path)
    validation_samples = read_from_list(validation_file, data_path)
    return train_samples, validation_samples


def get_train_and_validation_samples(root_path):
    data_path = osp.join(root_path, osp.join('data','train'))
    samples = get_all_samples(data_path, has_target=True)

    values = []
    for key, sample in samples.items():
        mask = read_mask(sample.mask_fp)
        mask_sum = np.sum(mask)/255.0
        values.append((mask_sum/101.0/101.0, key))
    values.sort()

    validation_samples = values[::5]
    validation_keys = [key for param, key in validation_samples]
    training_keys = [key for param, key in values if key not in validation_keys]

    validation_samples = {}
    for k in validation_keys:
        validation_samples[k] = samples[k]

    training_samples = {}
    for k in training_keys:
        training_samples[k] = samples[k]

    return training_samples, validation_samples

from augmentation import *

def test_augment(image):
    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101

    image = do_pad(image, DY0, DY1, DX0, DX1)
    return image

def validation_augment(image,mask):

    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101

    image, mask = do_pad2(image, mask, DY0, DY1, DX0, DX1)
    return image,mask

def train_augment(image,mask):

    DY0, DY1, DX0, DX1 = compute_center_pad(101, 101, factor=32)
    Y0, Y1, X0, X1 = DY0, DY0 + 101, DX0, DX0 + 101

    if np.random.rand() < 0.5:
         image, mask = do_horizontal_flip2(image, mask)
    #
    if np.random.rand() < 0.2:
        c = np.random.choice(2)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)
        if c==1:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0,0.1) )
        if c==2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0,dy=0,scale=1,angle=np.random.uniform(0,10))

    image, mask = do_pad2(image, mask, DY0, DY1, DX0, DX1)
    return image,mask

class TGSSaltDataset(Dataset):
    def __init__(self, samples, phase='train'):
        self.samples = samples
        self.keys = list(self.samples.keys())
        self.phase = phase
        if self.phase == 'train':
            random.shuffle(self.keys)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        if idx >= len(self.keys):
            idx = random.randint(0, len(self.keys)-1)

        sample = self.samples[self.keys[idx]]

        image = read_image(sample.image_fp)
        mask = read_mask(sample.mask_fp)

        image = np.array(image)/255.0
        mask = np.array(mask)/255.0

        if self.phase == 'train':
            image, mask = train_augment(image, mask)
        else:
            image, mask = validation_augment(image, mask)

        image = np.float32(image)
        mask = np.float32(mask)

        image = self.to_tensor(image)
        image = self.normalize(image)

        mask = self.to_tensor(mask[:,:,None])[0]

        return image, mask

    def __len__(self):
        # if self.phase == 'train':
        #     return len(self.keys) * 10
        return len(self.keys)

class TGSSaltDatasetTest(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.keys = list(self.samples.keys())

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        sample = self.samples[self.keys[idx]]

        image = read_image(sample.image_fp)
        image = np.array(image, dtype=np.float32)/255.0
        image = test_augment(image)

        image = self.to_tensor(image)
        image = self.normalize(image)

        return image

    def __len__(self):
        return len(self.keys)
