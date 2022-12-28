import cv2
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

import scipy.stats as ss

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm

from PIL import Image, ImageDraw, ImageFont
import PIL

import os

import copy
import random
import time


def crop_rotate(background, image, alpha, output_size=None, rotate_background=False):
    """
    Pastes an image onto the background, 
    rotates it and crops to leave no black pixels in corners.
    Background (NxN image) - has to be big enough for this operation
    """
    alpha_rad = alpha / 180 * np.pi
    back = background.copy()
    img = image.copy()

    back = back.rotate(np.random.randint(180) * int(rotate_background))
    back.paste(img, ((back.size[0] - img.size[0]) // 2, (back.size[1] - img.size[1]) // 2))
    back = back.rotate(alpha)

    a = np.ceil(np.sqrt(image.size[0] ** 2 + image.size[1] ** 2))
    b = (background.size[0] - a) // 2
    back = back.crop((b, b, b + a + 1, b + a + 1))
    return back


class Generator:
    def __init__(self, n_classes=10, batch_size=32, model_type='classifier', min_bound=-30, max_bound=30, size=224, blur=0, preprocessor=None, PREFIX=""):
        self.len = 0
        self.bounds = [min_bound, max_bound]
        self.n_classes = n_classes
        self.model_type = model_type
        self.picture_size = size
        self.batch_size = batch_size
        self.angles = []
        self.blur = blur
        self.preprocessor = preprocessor
        self.gen = None
        self.PREFIX = PREFIX

    def crop_rotate(self, background, image, alpha):
        """
        Pastes an image onto the background, 
        rotates it and crops to leave no black pixels in corners.
        Background (NxN image) - has to be big enough for this operation
        """
        alpha_rad = alpha / 180 * np.pi
        back = background.copy()
        img = image.copy()

        back = back.rotate(np.random.randint(180) * int(self.rotate_background))
        back.paste(img, ((back.size[0] - img.size[0]) // 2, (back.size[1] - img.size[1]) // 2))
        back = back.rotate(alpha)

        a = np.ceil(np.sqrt(image.size[0] ** 2 + image.size[1] ** 2))
        b = (background.size[0] - a) // 2
        back = back.crop((b, b, b + a + 1, b + a + 1))
        return back

    def transform(self, image_path, background, alpha):
        img = PIL.Image.open(image_path)
        img = self.crop_rotate(background, img, alpha=alpha).resize((self.picture_size, self.picture_size))
        np_img = np.array(img)
        if self.blur:
            np_img = cv2.GaussianBlur(np_img, (self.blur, self.blur), 0)
        return np.array([
            np_img[:, :, 0].reshape(np_img.shape[0], np_img.shape[1]), 
            np_img[:, :, 1].reshape(np_img.shape[0], np_img.shape[1]), 
            np_img[:, :, 2].reshape(np_img.shape[0], np_img.shape[1])
        ])

    def make_generator(self, backgrounds, images):
        if self.model_type == 'classifier':
            self.angles = np.linspace(self.bounds[0], self.bounds[1], self.n_classes)
        else:
            self.angles = np.arange(self.bounds[0], self.bounds[1] + 1)
        for n in range(len(images) // self.batch_size if len(images) % self.batch_size == 0 else len(images) // self.batch_size + 1):
            batch_x = []
            batch_y = []
            for i in range(min(self.batch_size, len(images) - n * self.batch_size)):
                y = np.random.randint(len(self.angles))
                alpha = self.angles[y]
                k = np.random.randint(len(backgrounds))
                batch_x.append(self.transform(self.PREFIX + "dataset/images/" + images[self.batch_size * n + i], backgrounds[k], alpha))
                batch_y.append(y if self.model_type == "classifier" else alpha)
            batch_x = torch.from_numpy(np.array(batch_x)).type(torch.FloatTensor)
            if self.preprocessor:
                batch_x = self.preprocessor(batch_x)
            batch_y = torch.tensor(batch_y).type(torch.LongTensor)
            yield batch_x, batch_y

    def initialize(self, backgrounds, images, rotate_background=False):
        self.rotate_background = rotate_background
        self.gen = self.make_generator(backgrounds, images)
        self.len = len(images) // self.batch_size if len(images) % self.batch_size == 0 else len(images) // self.batch_size + 1

    def get_angles(self):
        return self.angles

    def get_prediction(self, i):
        return self.angles[i]

    def __len__(self):
        return self.len

    def __next__(self):
        if self.gen is None:
            raise Exception("Initialize the Generator first")
        return next(self.gen)

    def __iter__(self):
        if self.gen is None:
            raise Exception("Initialize the Generator first")
        return self.gen


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, gen, optimizer, criterion, device, augmentator=None):
    torch.manual_seed(1337)
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for x, y in tqdm(gen, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(gen), epoch_acc / len(gen)


def evaluate(model, gen, criterion, device, augmentator=None):

    epoch_loss = 0
    epoch_acc = 0
    epoch_l1 = 0
    epoch_n = 0

    model.eval()

    with torch.no_grad():

        for x, y in tqdm(gen, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_argmax = np.argmax(np.array(y_pred), axis=1)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_l1 += np.abs(y_argmax - np.array(y)).sum()
            epoch_n += x.shape[0]
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(gen), epoch_acc / len(gen), epoch_l1 / epoch_n
