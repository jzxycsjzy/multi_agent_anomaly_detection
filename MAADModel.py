# Basic lib
import os
import numpy as np
import time
from tqdm import tqdm

# Neural network lib
import torch
import torch.nn as nn

class MAADModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3)
        self.pool1 = nn.MaxPool1d(3, stride=1)
        self.conv2 = nn.Conv1d(1,1,3)
        self.pool2 = nn.MaxPool1d(3, stride=1)
        self.conv3 = nn.Conv1d(1,1,3)
        self.pool3 = nn.MaxPool1d(3, stride=1)
        self.norm = nn.BatchNorm1d(1)
        self.fcnet = nn.Linear(38, 15)
        self.softmax = nn.Softmax(2)


    def forward(self, input):
        output = self.conv1(input=input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.pool3(output)
        output = self.norm(output)
        classification = self.fcnet(output)
        # classification = self.softmax(classification)
        return output, classification.squeeze(dim=0)

def model_test():
    model = MAADModel()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    optimizer.zero_grad()
    start_time = time.time()
    tgt = [4]
    tgt = torch.tensor([4], dtype=torch.long)
    crossentropyloss = nn.CrossEntropyLoss()
    for i in tqdm(range(1000)):
        a = torch.rand([1,1,50], dtype=torch.float32)
        out, classification = model(a)
        loss = crossentropyloss(classification, tgt)
        # loss = sum([abs(cla[i] - tgt[i]) for i in range(15)])
        # loss = torch.sum(out, dim=2) - torch.ones([1,1], dtype=torch.float32)
        print(classification)
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    time_consuming = end_time - start_time
    print("Time cost:{}".format(time_consuming))

if __name__ == '__main__':
    model_test()