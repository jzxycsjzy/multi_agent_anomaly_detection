# Basic lib
import os
import numpy as np
import time
from tqdm import tqdm

# Neural network lib
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
# https://pan.quark.cn/s/c469cc40636c#/list/share

class MAADModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 150
        self.lstm = nn.LSTM(input_size=150, hidden_size=self.hidden_size, num_layers=7)
        self.out_category = nn.AdaptiveAvgPool2d((1, 15))
        self.out_feature = nn.Linear(self.hidden_size, 150)
        # self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)

        # self.conv1_span = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool1_span = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv2_span = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool2_span = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv3_span = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool3_span = nn.MaxPool2d(3, stride=1, padding=1)
        # # self.conv4 = nn.Conv2d(1, 1, 3, padding=1)
        # # self.pool4 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.norm = nn.BatchNorm2d(1)
        # self.fcnet = nn.AdaptiveAvgPool2d((1, 15))


    def forward(self, input: torch.tensor, span_input: torch.tensor):
        if input != None:
            span_input = torch.cat([span_input, input], dim=1)
        out, hidden = self.lstm(span_input)
        out_feature = self.out_feature(out[:, -1, :])
        out_feature = out_feature.unsqueeze(0)
        out_category = self.out_category(out)
        return out_feature, out_category
        # if input != None:
        #     output = self.conv1(input=input)
        #     output = self.pool1(output)
        #     output = self.conv2(output)
        #     output = self.pool2(output)
        #     output = self.conv3(output)
        #     output = self.pool3(output)

        # output_span = self.conv1_span(input=span_input)
        # output_span = self.pool1_span(output_span)
        # output_span = self.conv2_span(output_span)
        # output_span = self.pool2_span(output_span)
        # output_span = self.conv3_span(output_span)
        # output_span = self.pool3_span(output_span)
        # # output = self.conv4(output)
        # # output = self.pool4(output)
        # # Feature concat
        # output = torch.cat([output, output_span], dim=2) if input !=None else output_span
        # output = self.norm(output)
        # print(output.shape)
        # classification = self.fcnet(output)
        # # classification = nn.functional.adaptive_avg_pool2d(output, output_size=(1,15))
        # # classification = classification.squeeze(0).squeeze(0)
        # # classification = self.fcnet(output)
        # # classification = self.softmax(classification)
        # return output, classification.squeeze(0).squeeze(0)

class MAADCategory(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        # self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.ln1 = nn.Linear(1, 3)
        self.ac1 = nn.ReLU(inplace=False)
        self.ln2 = nn.Linear(3, 5)
        self.ac2 = nn.ReLU(inplace=False)
        self.ln3 = nn.Linear(5, 7)
        self.ac3 = nn.ReLU(inplace=False)

        self.fcnet = nn.AdaptiveAvgPool2d((1, 15))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        output = self.ln1(input)
        output = self.ac1(output)
        output = self.ln2(output)
        output = self.ac2(output)
        output = self.ln3(output)
        output = self.ac3(output)
        # output = self.conv1(input=input)
        # output = self.pool1(output)
        # output = self.conv2(output)
        # output = self.pool2(output)
        # output = self.conv3(output)
        # output = self.pool3(output)
        # output = self.relu(output)

        output = self.fcnet(output)
        output = self.softmax(output)

        return output

def model_test():
    model = MAADModel()
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    optimizer.zero_grad()
    start_time = time.time()
    # tgt = torch.tensor([4], dtype=torch.long)
    # crossentropyloss = nn.CrossEntropyLoss()
    # for i in tqdm(range(1000000)):
    #     a = torch.ones([1, 1, 1, 50], dtype=torch.float32)
    #     out, classification = model(a)
    #     # print(out, classification)
    #     loss = crossentropyloss(classification, tgt)
    #     # loss = sum([abs(cla[i] - tgt[i]) for i in range(15)])
    #     # loss = torch.sum(out, dim=2) - torch.ones([1,1], dtype=torch.float32)
    #     if i % 1000 == 0:
    #         print(loss, classification)
    #     loss.backward()
    #     optimizer.step()
        
    end_time = time.time()
    time_consuming = end_time - start_time
    print("Time cost:{}".format(time_consuming))

if __name__ == '__main__':
    a = torch.tensor(1, dtype=torch.float32)
    b = torch.tensor(2, dtype=torch.float32)
    c = torch.tensor(3, dtype=torch.float32)
    d = torch.tensor([a,b,c])
    print(d)
    print(torch.mean(d))
    # model = MAADModel()
    # optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    # optimizer.zero_grad()
    # tgt = torch.tensor([4], dtype=torch.long)
    # crossentropyloss = nn.CrossEntropyLoss()
    # for i in range(1):
    #     sample = torch.ones([1, 50, 150], dtype=torch.float32)
    #     sample_span = torch.ones([1, 50, 150], dtype=torch.float32)
    #     out, category = model(sample, sample_span)
    #     print(out.shape, category.shape)
    #     out = torch.cat([out] * 7 , dim=1)
    #     category_model = MAADCategory()
    #     out = category_model(out)
    #     out = out.squeeze(0)
    #     print(out.shape)
    #     loss = crossentropyloss(out, tgt)
    #     print(loss)

