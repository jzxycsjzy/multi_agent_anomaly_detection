# Basic lib
import os
import numpy as np
import time
from tqdm import tqdm

# Neural network lib
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

class MAADModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool4 = nn.MaxPool2d(3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(1)
        self.fcnet = nn.AdaptiveAvgPool2d((1, 15))


    def forward(self, input):
        output = self.conv1(input=input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.pool3(output)
        # output = self.conv4(output)
        # output = self.pool4(output)
        output = self.norm(output)
        classification = self.fcnet(output)
        # classification = nn.functional.adaptive_avg_pool2d(output, output_size=(1,15))
        # classification = classification.squeeze(0).squeeze(0)
        # classification = self.fcnet(output)
        # classification = self.softmax(classification)
        return output, classification.squeeze(0).squeeze(0)

def model_test():
    model = MAADModel()
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    optimizer.zero_grad()
    start_time = time.time()
    tgt = torch.tensor([4], dtype=torch.long)
    crossentropyloss = nn.CrossEntropyLoss()
    for i in tqdm(range(1000000)):
        a = torch.ones([1, 1, 1, 50], dtype=torch.float32)
        out, classification = model(a)
        # print(out, classification)
        loss = crossentropyloss(classification, tgt)
        # loss = sum([abs(cla[i] - tgt[i]) for i in range(15)])
        # loss = torch.sum(out, dim=2) - torch.ones([1,1], dtype=torch.float32)
        if i % 1000 == 0:
            print(loss, classification)
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    time_consuming = end_time - start_time
    print("Time cost:{}".format(time_consuming))

if __name__ == '__main__':
    modela = nn.modules.Conv2d(1, 1, 3, padding=1)
    modela.cuda()
    modelb = nn.modules.Conv2d(1, 1, 3, padding=1)
    modelb.cuda()
    modelc = nn.AdaptiveAvgPool2d((1, 15))
    modelc.cuda()
    randinput = torch.rand((1, 1, 1, 50))
    randinput = randinput.to("cuda:0")
    out = modela(randinput)
    out = modelb(out)
    out = modela(out)
    out = modelb(out)
    cls = modelc(out)
    tgt = torch.tensor([3], dtype=torch.long)
    tgt = tgt.to("cuda:0")
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(cls, tgt)
    loss.backward()