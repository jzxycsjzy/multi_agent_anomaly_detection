# Basic lib
import os
import numpy as np
import time
from tqdm import tqdm
import random

# Neural network lib
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
torch.autograd.set_detect_anomaly(True)

# 引用实时， throughput, model update

class MAADModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(1)

        self.feature_compress = nn.LSTM(300, 300, 2)
        # self.category = nn.AdaptiveAvgPool2d((1, 72))
        self.category = nn.Linear(300, 72)


    def forward(self, input):
        # Feature fusion
        output = self.conv1(input=input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.pool3(output)
        # output = self.conv4(output)
        # output = self.pool4(output)
        output = self.norm(output)
        output = output.squeeze(0)
        output, _ = self.feature_compress(output)
        output = output[:, -1, :]
        classification = self.category(output)
        classification = torch.softmax(classification, dim=-1)
        # classification = nn.functional.adaptive_avg_pool2d(output, output_size=(1,15))
        # classification = classification.squeeze(0).squeeze(0)
        # classification = self.fcnet(output)
        # classification = self.softmax(classification)
        return output.unsqueeze(0).unsqueeze(0), classification
    
class CategoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(72, 72, 5)

        # self.fcnet1 = nn.Linear(150, 360)
        # self.fcnet2 = nn.Linear(360, 180)
        # self.fcnet3 = nn.Linear(180, 90)
        self.fcnet4 = nn.Linear(72, 72)

    def forward(self, x):
        x = x.float()
        out, _ = self.lstm(x)
        out = out[:,-1, :]
        # out = self.fcnet1(out)
        # out = self.fcnet2(out)
        # out = self.fcnet3(out)
        out = self.fcnet4(out)
        out = torch.softmax(out, dim=-1)
        return out
