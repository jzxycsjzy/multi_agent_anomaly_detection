# Basic lib
import os
import numpy as np
import time
from tqdm import tqdm

# Neural network lib
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.autograd.set_detect_anomaly(True)

class MAADModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Span network
        self.conv1_span = nn.Conv2d(1, 3, 3, padding=1)
        self.pool1_span = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm1_span = nn.BatchNorm2d(3)

        self.conv2_span = nn.Conv2d(3, 3, 3, padding=1)
        self.pool2_span = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm2_span = nn.BatchNorm2d(3)

        self.conv3_span = nn.Conv2d(3, 1, 3, padding=1)
        self.pool3_span = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm3_span = nn.BatchNorm2d(1)
        # Log network
        self.conv1_log = nn.Conv2d(1, 3, 3, padding=1)
        self.pool1_log = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm1_log = nn.BatchNorm2d(3)

        self.conv2_log = nn.Conv2d(3, 3, 3, padding=1)
        self.pool2_log = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm2_log = nn.BatchNorm2d(3)

        self.conv3_log = nn.Conv2d(3, 1, 3, padding=1)
        self.pool3_log = nn.AvgPool2d(3, stride=1, padding=1)
        self.norm3_log = nn.BatchNorm2d(1)

        # self.conv4 = nn.Conv2d(5, 3, 3, padding=1)
        # self.pool4 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.norm4 = nn.BatchNorm2d(3)

        # self.conv5 = nn.Conv2d(3, 1, 3, padding=1)
        # self.pool5 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.norm5 = nn.BatchNorm2d(1)
        # self.feature_compress = nn.Linear(300, 15)

        self.log_feature_compress = nn.LSTM(300, 600, 2)
        self.span_feature_compress = nn.LSTM(300, 600, 2)

        # self.classifier = nn.AdaptiveMaxPool2d((1, 15))
        self.classifier = nn.Linear(600, 2)
        


    def forward(self, log_input: torch.tensor, span_input: torch.tensor):
        # if input != None:
        #     # Feature fusion
        #     span_input = torch.cat([span_input, input], dim=-2)
        # # Forward
        # span_output = self.conv1_span(span_input)
        # span_output = self.pool1_span(span_output)
        # span_output = self.norm1_span(span_output)
        # span_output = torch.relu(span_output)
        # span_output = self.conv2_span(span_output)
        # span_output = self.pool2_span(span_output)
        # span_output = self.norm2_span(span_output)
        # span_output = torch.relu(span_output)
        # span_output = self.conv3_span(span_output)
        # span_output = self.pool3_span(span_output)
        # span_output = self.norm3_span(span_output)
        # span_output = torch.relu(span_output)
        span_output = span_input.squeeze(0)
        span_output, _ = self.span_feature_compress(span_output)
        span_output = span_output[:, -1, :]
        
        # if log_input != None:
        #     log_output = self.conv1_log(log_input)
        #     log_output = self.pool1_log(log_output)
        #     log_output = self.norm1_log(log_output)
        #     log_output = torch.relu(log_output)
        #     log_output = self.conv2_log(log_output)
        #     log_output = self.pool2_log(log_output)
        #     log_output = self.norm2_log(log_output)
        #     log_output = torch.relu(log_output)
        #     log_output = self.conv3_log(log_output)
        #     log_output = self.pool3_log(log_output)
        #     log_output = self.norm3_log(log_output)
        #     log_output = torch.relu(log_output)
        #     log_output = log_output.squeeze(0)
        #     log_output, _ = self.log_feature_compress(log_output)
        #     log_output = log_output[:, -1, :]
        
        if log_input != None:
            # Feature fusion-mfb
            log_output = log_input.squeeze(0)
            log_output, _ = self.log_feature_compress(log_output)
            log_output = log_output[:, -1, :]
            iq = torch.mul(span_output, log_output)
            iq = F.dropout(iq, 0.5, training=self.training)
        else:
            iq = span_output
        iq = iq.view(-1, 1, 1, 600)
        dt = F.normalize(iq)
        dt = self.conv1_span(dt)
        dt = self.pool1_span(dt)
        dt = self.norm1_span(dt)
        dt = torch.relu(dt)
        dt = self.conv2_span(dt)
        dt = self.pool2_span(dt)
        dt = self.norm2_span(dt)
        dt = torch.relu(dt)
        dt = self.conv3_span(dt)
        dt = self.pool3_span(dt)
        dt = self.norm3_span(dt)
        dt = torch.relu(dt)
        dt = dt.squeeze(0)
        # dt, _ = self.span_feature_compress(dt)
        # dt = dt[:, -1, :]
        category = self.classifier(dt)   
        category = torch.softmax(category, -1)
        if len(iq.shape) != 2:
            # iq = iq.squeeze(0).squeeze(0)
            category = category.squeeze(0)
        return category
        # output = self.conv1(log_input)
        # output = self.pool1(output)
        # output = self.norm1(output)
        # output = self.conv2(output)
        # output = self.pool2(output)
        # output = self.norm2(output)
        # output = self.conv3(output)
        # output = self.pool3(output)
        # output = self.norm3(output)
        # output = self.conv4(output)
        # output = self.pool4(output)
        # output = self.norm4(output)
        # output = self.conv5(output)
        # output = self.pool5(output)
        # output = self.norm5(output)
        # output = output.squeeze(0)
        # # print(output.shape)
        # # features = self.feature_compress(output)
        # # print(features)
        # features, hidden = self.feature_compress(output)
        # features = features[:, -1, :]
        # classification = self.classifier(features)
        # classification = torch.softmax(classification, dim=-1)
        # return features.unsqueeze(0).unsqueeze(0), classification

class MAADCategory(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 5, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(5, 5, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(5, 3, 3, padding=1)
        self.pool4 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(3, 1, 3, padding=1)
        self.pool5 = nn.MaxPool2d(3, stride=1, padding=1)
        # self.relu = nn.ReLU(inplace=False)
        self.norm = nn.BatchNorm2d(1)
        self.lstm = nn.LSTM(300, 300, 2)
        self.classifier = nn.Linear(300, 71)
        # self.lstm = nn.LSTM(input_size=150, hidden_size=100, num_layers=7)
        # self.out_category = nn.AdaptiveMaxPool2d((1, 72))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        output = self.conv1(input=input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.pool3(output)
        output = self.conv4(output)
        output = self.pool4(output)
        output = self.conv5(output)
        output = self.pool5(output)
        output = self.norm(output)
        output = output.squeeze(0)
        print(output.shape)
        out, hidden = self.lstm(output)
        out = out[:, -1, :]
        output = self.classifier(out)
        output = self.softmax(output)

        return output


class DecisionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcnet_category = nn.Linear(in_features=2520, out_features=300)
        self.fcnet_features = nn.Linear(in_features=10500, out_features=300)

        self.conv1 = nn.Conv2d(1, 1, 2)
        self.fcnet = nn.Linear(in_features=299, out_features=72)
        # self.classifier = nn.AdaptiveMaxPool2d((1, 15))

    def forward(self, category, features):
        out1 = self.fcnet_category(category)
        out2 = self.fcnet_features(features)
        out = torch.cat([out1, out2], dim=-2)
        out = out.unsqueeze(0).unsqueeze(0)
        out = self.conv1(out)
        # out = self.fcnet2(out)
        out = torch.softmax(out, dim=-1)
        return out.squeeze(0).squeeze(0)


def model_test():
    model = MAADModel()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    optimizer.zero_grad()
    start_time = time.time()
    tgt = torch.tensor([4], dtype=torch.long)
    crossentropyloss = nn.CrossEntropyLoss()
    for i in tqdm(range(1)):
        a = torch.rand([1, 1, 8, 300], dtype=torch.float32)
        b = torch.rand([1, 1, 9, 300], dtype=torch.float32)
        classification = model(None, a)
        print(classification.shape)
        loss = crossentropyloss(classification, tgt)
        # loss = sum([abs(cla[i] - tgt[i]) for i in range(15)])
        # loss = torch.sum(out, dim=2) - torch.ones([1,1], dtype=torch.float32)
        if i % 10 == 0:
            print(loss, classification)
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    time_consuming = end_time - start_time
    print("Time cost:{}".format(time_consuming))

if __name__ == '__main__':
    # model_test()
    tgt_file = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/F01-01/F0101raw_log2021-08-14_10-22-51.log"
    with open(tgt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        info = line.split(' ')
        info = info[2].split(',')[2]
        print(info)
        break
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

