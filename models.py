import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data


class CNNModelOneConv(nn.Module):
    def __init__(self, output_dim=360):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 2 * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(75264, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim)
        )
                
    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(-1, 75264)
        x = self.linear_block(x)
        
        return x

    def batch_predict(self, x):
        logits = self.forward(x.transpose(2, 3).transpose(1, 2))
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


class CNNModelTwoConv(nn.Module):
    def __init__(self, output_dim=360):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 3 * 3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(3 * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * 3, 9 * 3, kernel_size=3, stride=3, padding=2),
            nn.BatchNorm2d(9 * 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(38 * 38 * 9 * 3, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim)
        )
                
    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(-1, 38 * 38 * 9 * 3)
        x = self.linear_block(x)
        
        return x

    def batch_predict(self, x):
        logits = self.forward(torch.from_numpy(x).type(torch.FloatTensor).transpose(2, 3).transpose(1, 2))
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()