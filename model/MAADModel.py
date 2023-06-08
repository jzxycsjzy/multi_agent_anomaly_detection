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

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
torch.autograd.set_detect_anomaly(True)

class MAADModel(nn.Module):
    def __init__(self, error_types: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(1)

        self.feature_compress = nn.LSTM(300, 300, 2)
        self.category = nn.Linear(300, error_types)


    def forward(self, input):
        # Feature fusion
        output = self.conv1(input=input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.pool3(output)
        output = self.norm(output)
        output = output.squeeze(0)
        output, _ = self.feature_compress(output)
        output = output[:, -1, :]
        classification = self.category(output)
        classification = torch.softmax(classification, dim=-1)
        # classification = nn.functional.adaptive_avg_pool2d(output, output_size=(1,15))
        return output.unsqueeze(0).unsqueeze(0), classification
