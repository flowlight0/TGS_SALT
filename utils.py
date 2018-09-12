import os
import os.path as osp

import cv2
import torch
import shutil

class Sample:
    def __init__(self):
        self.image_fp = None
        self.mask_fp = None

def get_file_id(filepath):
    file_name = filepath.split(os.sep)[-1]
    id = file_name.split('.')[0]

    return id

def read_bad_samples():
    with open('badsamples.txt', 'r') as file:
        samples = file.readlines()
        samples = [sample[:-1] for sample in samples]
    return samples

def net_compatible_image_size(img):
    height, width, _ = img.shape

    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT, 0)
    return img


def read_image(file):
    img = cv2.imread(str(file))
    return img

def read_mask(file):
    mask = read_image(file)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return mask


def save_checkpoint(state, data_path, is_best, suffix=''):
    filepath = osp.join(data_path, 'checkpoint'+suffix+'.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, osp.join(data_path, 'model_best'+suffix+'.pth.tar'))