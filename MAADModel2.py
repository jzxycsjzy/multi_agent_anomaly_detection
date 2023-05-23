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

import matplotlib.pyplot as plt
def writeGraph(x, y, x_label, y_label, title):
    plt.figure(figsize=(7.5, 3))
    plt.plot(x, y, markerfacecolor='blue', marker='o')
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.xlim((0, 510))
    plt.ylim((0,0.04))
    plt.xticks(size=15)
    plt.yticks([0, 0.02, 0.04], size=15)
    plt.gca().set_aspect(3000)
    # plt.title(title, fontsize=15)
    plt.savefig(title + ".png")
    plt.clf()

def model_test():
    # tgt_files = ["/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/F01-01/F0101raw_log2021-08-14_10-22-51.log", "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/normal/F02-04-normal/raw_log.log", "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/normal/normal0809_03/raw_log2021-08-10_12-10-15.log", "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/normal/normal0822_02/normal2_2021-08-22_22-05-06.log"]
    
    # all_lines = []
    # for tgt_file in tgt_files:
    #     with open(tgt_file, 'r') as f:
    #         all_lines += f.readlines()
    # dt = {}
    # for i in tqdm(range(len(all_lines))):
    #     line = all_lines[i]
    #     info = line.split(' ')
    #     info = info[2].split(',')[2]
    #     if info in dt.keys():
    #         dt[info] += 1
    #     else:
    #         dt[info] = 0
    # sum = 0
    # for key in dt.keys():
    #     sum += dt[key]
    # print(sum / len(dt.keys()))
    model = MAADModel()
    log_list = []
    time_list = []
    log_sim = torch.rand([1, 1, 10, 300], dtype=torch.float32)
    model(log_sim)
    for i in tqdm(range(0, 11)):
        log_length = i * 50
        if i == 0:
            log_length = 1
        log_sim = torch.rand([1, 1, log_length, 300], dtype=torch.float32)
        start_time = time.time()
        classification = model(log_sim)
        end_time = time.time()
        log_list.append(log_length)
        time_list.append(end_time - start_time)
    writeGraph(log_list, time_list, "Number of Logs", "Latency(s)", "Single agent time latency")
    #     # print(classification.shape)
        # loss = crossentropyloss(classification, tgt)
        # # loss = sum([abs(cla[i] - tgt[i]) for i in range(15)])
        # # loss = torch.sum(out, dim=2) - torch.ones([1,1], dtype=torch.float32)
        # if i % 10 == 0:
        #     print(loss, classification)
        # loss.backward()
        # optimizer.step()
    # parameter_list = []
    # # Create drain template miner.
    # data_dir = "./data/DeepTraLog/TraceLogData"

    # tgt_file = ""
    # # Read All data
    # for f_0 in os.listdir(data_dir):
    #     log_file = ""
    #     trace_files = []
    #     f_1 = os.path.join(data_dir, f_0)
    #     for f_2 in os.listdir(f_1):
    #         f_3 = os.path.join(f_1, f_2)
    #         # If has sub folder
    #         if os.path.isdir(f_3):
    #             tf = []
    #             lf = ""
    #             for f_4 in os.listdir(f_3):
    #                 f_5 = os.path.join(f_3, f_4)
    #                 if f_4.endswith("csv"):
    #                     tf.append(f_5)
    #                 else:
    #                     lf = f_5
    #             parameter_list.append(lf)
    #             # p = Process(target=TraceLogCombine, args=(drain3_template, tf, lf, tgt_file))
    #             # process_list.append(p)
    #             # TraceLogCombine(drain3_template, tf, lf, tgt_file)
    #         else:
    #             # If has no sub folder
    #             if f_3.endswith("csv"):
    #                 trace_files.append(f_3)
    #             else:
    #                 log_file = f_3
    #     if log_file != "":
    #         # TraceLogCombine(drain3_template, trace_files, log_file, tgt_file)
    #         parameter_list.append(log_file)
    #         # p = Process(target=TraceLogCombine, args=(drain3_template, tf, lf, tgt_file))
    #         # process_list.append(p)

    # all_lines = []
    # for tgt_file in parameter_list:
    #     with open(tgt_file, 'r') as f:
    #         all_lines += f.readlines()
    # dt = {}
    # for i in tqdm(range(len(all_lines))):
    #     line = all_lines[i]
    #     info = line.split(' ')
    #     info = info[2].split(',')[2]
    #     if info in dt.keys():
    #         dt[info] += 1
    #     else:
    #         dt[info] = 0
    # sum = 0
    # for key in dt.keys():
    #     sum += dt[key]
    # print(sum / len(dt.keys()))


if __name__ == '__main__':
    model_test()