from __future__ import annotations
from typing import Tuple

from workflow import Init_workflow, Logs2Vectors

import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

#Import torch
import torch
from torch.optim import Adam

# from memory_profiler import profile

from drain3 import TemplateMiner
from MAADModel import MAADModel
from MAADModel import DecisionFusion as MAADCategory
from data_preparation import Drain_Init, RemoveSignals

# Import NLP model
from SIF.src import params, data_io, SIF_embedding

__stdout__ = sys.stdout # 标准输出就用这行
open("log_test.txt", 'w+')
sys.stdout = open('log_test.txt', 'a')

# Init SIF parameters
wordfile = "data/glove/vectors.txt" # word vector file, can be downloaded from GloVe website
weightfile = "data/glove/vocab.txt" # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# set parameters
param = params.params()
param.rmpc = rmpc

total_tensor = []
softmax = torch.nn.Softmax(dim=1)

cur_seq = []
cur_faults = []

def Init_model() -> Tuple[dict[str: MAADModel], MAADCategory]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    optimizer_list = {}
    service_list = pd.read_csv("id_service.csv").drop(labels="Unnamed: 0", axis=1)
    # service_list = service_list.sample(frac=1).reset_index(drop=True)
    # print(service_list.loc[0])
    category_model = MAADCategory()
    category_model.load_state_dict(torch.load("/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/models/Model2_00/" + "category.pt"))
    for i in tqdm(range(service_list.shape[0])):
        service = service_list.loc[i]["Service"]
        model_list[service] = MAADModel()
        model_list[service].load_state_dict(torch.load("/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/models/Model2_00/" + service + ".pt"))
        model_list[service].eval()
    return model_list, category_model

def Init_workflow() -> dict:
    fault_list = pd.read_csv("id_fault.csv", index_col="id")
    return fault_list.to_dict()['filename']

def Sample(fault_list):
    """
    Sample from each data frame and train the models
    """
    batch_size = 16
    model_dict, category = Init_model()

    tmp = Drain_Init()
    tmp.load_state("tt2")
    
    start_time = time.time()

    data_dir = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/ProcessedData3/test/"
    file_list = os.listdir(data_dir)

    for epoch in range(15):
        print("Epoch {} has started.".format(epoch))
        pd_file_list = [[] for i in range(batch_size)]
        batch_count = 0
        correct = 0
        for i in tqdm(range(len(file_list))):
            fault_type = int(file_list[i].split('_')[1].split('.')[0])
            file = os.path.join(data_dir, file_list[i])
            # cur_pd = pd.read_csv(file)
            pd_file_list[batch_count].append(file)
            res = ProcessData(model_dict, category, file, fault_list, tmp, fault_type)
            print(res)
            if res == fault_type:
                print("correct")
                correct += 1
            if i % 100 == 0 and i != 0:
                print("##############")
                print(correct, i)


def ProcessData(models: dict[str: MAADModel], category_model: MAADCategory, file: str, fault_list: dict, tmp: TemplateMiner, fault_type: int):
    cur_pd = pd.read_csv(file)
    (path, file_name) = os.path.split(file)
    fault_type = int(file_name.split('_')[1].split('.')[0])
    category_list = []
    feature_list = []
    WorkTraceNew(models=models, optimizers=None, pd=cur_pd, fault_list=fault_list, tmp=tmp, fault_type=fault_type, category_list=category_list, feature_list=feature_list, tgt=None, category_model=category_model, category_optimizer=None)
    # is_normal = 1 if fault_type == 71 else 0
    # Result get
    return category_list[0]
    if res.index(max(res)) == fault_type:
        print("correct")


# def Sample(models: dict[str: MAADModel], pds: list[pd.DataFrame], fault_list: dict):
#     tmp = Drain_Init()
#     tmp.load_state("tt2")
#     steps = [0] * 15
#     traces = 0

#     total_times = 0
#     correct = 0

#     # Cal_times = random.randint(1,7)
#     Cal_times = 1
#     while True:
#         status = True
#         for i in range(15):
#             if pds[i].shape[0] > steps[i]:
#                 status = False
                
#                 break
#         if status:
#             steps = [0] * 15
#             break
#         for i in range(15):
#             # if i == 5 or i == 7:
#             #     continue
#             cur_pd = pds[i]
#             count = 0
#             start_index = steps[i]
#             end_index = steps[i]
#             while steps[i] < cur_pd.shape[0]:
#                 if cur_pd.loc[steps[i]]['parentspan'] == '-1' or cur_pd.loc[steps[i]]['parentspan'] == -1 or steps[i] == cur_pd.shape[0]:
#                     count += 1
#                     if count == 2:
#                         end_index = steps[i]
#                         traces += 1
#                         if traces % Cal_times == 0:
#                             WorkTrace(models=models, pd=cur_pd.iloc[start_index: end_index], fault_list=fault_list, tmp=tmp, fault_type=i)
#                             votes = [0] * 15
#                             max_confidence = 0
#                             max_confidents_vote = 0
#                             for category in cur_faults:
#                                 res = category.cpu().detach().numpy().tolist()[0]
#                                 if max(res) > max_confidence:
#                                     max_confidence = max(res)
#                                     max_confidents_vote = res.index(max(res))
#                                 votes[res.index(max(res))] += 1
#                             cur_faults.clear()
#                             votes[max_confidents_vote] += 13
#                             print(votes)
#                             print(max_confidents_vote)
#                             total_times += 1
#                             if votes.index(max(votes)) == i:
#                                 correct += 1
#                             if total_times % 100 == 0:
#                                 print("#################")
#                                 print(total_times, correct)
#                             total_tensor.clear()
#                             # Cal_times = random.randint(1,7)
#                             cur_seq.clear()
#                         count = 0
#                         break
#                 end_index = steps[i]
#                 steps[i] += 1

# def WorkTrace(models: dict[str: MAADModel], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int):
#     pd = pd.reset_index(drop=True)
#     # Obtain log and span vector
#     start_log = pd.loc[0]['loginfo']
#     start_span = pd.loc[0]['spaninfo']
#     cur_vectors = Logs2Vectors(start_log, tmp=tmp)
#     cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
#     # Make sure the service name
#     cur_service = pd.loc[0]['service']
#     print(fault_list[fault_type], pd.loc[0]['traceid'])
#     cur_tensor = None
#     if cur_vectors != []:
#         cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
#         cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
#     cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
#     cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)

#     cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)
#     cur_childs = eval(pd.loc[0]['childspans'])
#     if len(cur_childs) != 0:
#         for child in cur_childs:
#             child_series = pd[pd['spanid'] == child].reset_index(drop=True)
#             WorkForward(models=models, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
#     else:
#         cur_seq.append(cur_out)
#         cur_faults.append(category)


# def WorkForward(models: dict[str: MAADModel],  pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, cur_spans: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor):
#     cur_vectors = Logs2Vectors(cur_logs=cur_logs, tmp=tmp)
#     cur_span_vectors = Logs2Vectors(cur_logs=[cur_spans], tmp=tmp)
#     cur_out = prev_out

#     cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
#     if cur_tensor != None:
#         cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
#         cur_tensor = torch.cat([prev_out, cur_tensor], dim=2)
#     else:
#         cur_tensor = prev_out
#     cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
#     cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)
#     cur_span_vectors = torch.cat([prev_out, cur_span_tensor], dim=2)
#     cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)

#     cur_childs = eval(childs)
#     if len(cur_childs) != 0:
#         for child in cur_childs:
#             child_series = pd[pd['spanid'] == child].reset_index(drop=True)
#             WorkForward(models=models, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
#     # It could represent that this span has come to end. Calculate the loss and backward
#     else:
#         cur_seq.append(cur_out)
#         cur_faults.append(category)
def WorkTrace(models: dict[str: MAADModel], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int, category_list: list, feature_list: list):
    pd = pd.reset_index(drop=True)
    # Obtain log and span vector
    start_log = pd.loc[0]['loginfo']
    start_span = pd.loc[0]['spaninfo']
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
    # Make sure the service name
    cur_service = pd.loc[0]['service']
    cuda = next(models[cur_service].parameters()).device
    print(fault_type, fault_list[fault_type], pd.loc[0]['traceid'])
    cur_tensor = None
    if cur_vectors != []:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)

    cur_tensor = cur_tensor.to(cuda) if cur_tensor != None else None
    cur_span_tensor = cur_span_tensor.to(cuda)
    cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)
    # cur_out = None
    # cur_category = None
    # has_cal = False
    # res = False
    # if len(cur_vectors) != 0:
    #     cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32)
    #     cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
    #     cur_out, category = models[cur_service](cur_tensor)
    #     has_cal = True

    cur_childs = eval(pd.loc[0]['childspans'])
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            if child_series.shape[0] != 0:
                WorkForward(models=models, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out, category_list=category_list, feature_list=feature_list)
    else:
        # It could represent that this span has come to end. Calculate the loss and backward
        feature_list.append(cur_out)
        category_list.append(category)
        # tgt = torch.tensor([fault_type], dtype=torch.long)
        # loss = loss_func(category, tgt)
        # loss.backward(retain_graph=True)
        # for service in optimizers.keys():
        #     optimizers[service].step()
        #     # optimizers[service].zero_grad()
        # print(loss)


def WorkForward(models: dict[str: MAADModel], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, cur_spans: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor, category_list: list, feature_list: list):
    cuda = next(models[cur_service].parameters()).device
    
    cur_vectors = Logs2Vectors(cur_logs=cur_logs, tmp=tmp)
    cur_span_vectors = Logs2Vectors(cur_logs=[cur_spans], tmp=tmp)
    cur_out = prev_out

    cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
    if cur_tensor != None:
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
        cur_tensor = cur_tensor.to(cuda)
        cur_tensor = torch.cat([prev_out, cur_tensor], dim=2)
    else:
        cur_tensor = prev_out
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)
    cur_span_tensor = cur_span_tensor.to(cuda)
    cur_span_vectors = torch.cat([prev_out, cur_span_tensor], dim=2)
    cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)

    cur_childs = eval(childs)
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            if child_series.shape[0] != 0:
                WorkForward(models=models, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out, category_list=category_list, feature_list=feature_list)
    # It could represent that this span has come to end. Calculate the loss and backward
    else:
        feature_list.append(cur_out)
        category_list.append(category)
        # tgt = torch.tensor([fault_type], dtype=torch.long)
        # loss = loss_func(category, tgt)
        # loss.backward(retain_graph=True)
        # for service in optimizers.keys():
        #     optimizers[service].step()
        #     # optimizers[service].zero_grad()
        # print(loss)

def cal_all(tensors: list[torch.tensor]):
    res = None
    res2 = None
    confident_list = []
    for t in tensors:
        cache = t.detach().numpy().tolist()[0]
        confident_list.append(cache.index(max(cache)))
        cache2 = softmax(t)
        res2 = cache2 if res2 == None else res + cache2
        res = t if res == None else t + res
    if res != None:
        res = res.detach().numpy().tolist()[0]
        res = res.index(max(res))
        res2 = res2.detach().numpy().tolist()[0]
        res2 = res2.index(max(res2))
    return res, res2, confident_list


def WorkTraceNew(models: dict[str: MAADModel], optimizers: dict[str: Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int, category_list: list, feature_list: list, tgt: torch.tensor, category_model: MAADCategory, category_optimizer: Adam):
    pd = pd.reset_index(drop=True)
    span_data_collection = {}
    log_data_collection = {}
    cuda = next(models["ts-order-service"].parameters()).device
    for span_idx in range(pd.shape[0]):
        dt = pd.loc[span_idx]
        service = dt['service']
        span = dt['spaninfo']
        log = dt['loginfo']
        if service in span_data_collection.keys():
            span_data_collection[service] += [span]
            logs = eval(log)
            logs = [x[2] for x in logs]
            log_data_collection[service] += logs
        else:
            span_data_collection[service] = [span]
            logs = eval(log)
            logs = [x[2] for x in logs]
            log_data_collection[service] = logs
    out_data = []
    res_list = []
    for service in models.keys():
        if service not in span_data_collection.keys():
            out = torch.zeros([1, 72], dtype=torch.float32)
            out = out.to(cuda)
            res_list.append(out)
            continue
        # (span_data_collection[service], log_data_collection[service])
        span_vector = Logs2Vectors(span_data_collection[service], tmp)
        log_vector = Logs2Vectors(log_data_collection[service], tmp)

        span_tensor = torch.tensor(span_vector, dtype=torch.float32)
        span_tensor = span_tensor.unsqueeze(0).unsqueeze(0)

        if len(log_vector) != 0:
            log_tensor = torch.tensor(log_vector, dtype=torch.float32)
            log_tensor = log_tensor.unsqueeze(0).unsqueeze(0)
        else:
            log_tensor = None
        span_tensor = span_tensor.to(cuda)
        log_tensor = log_tensor.to(cuda) if log_tensor != None else None
        category = models[service](log_tensor, span_tensor)
        res_list.append(category)
        res = category.cpu().detach().numpy().tolist()[0]
        res = res.index(max(res))
        # loss = loss_func(category, tgt)
        # loss.backward()
        # optimizers[service].step()
        # optimizers[service].zero_grad()
        out_data.append([fault_type, res])
    category_tensor = torch.cat(res_list, dim=-1)
    result = category_model(category_tensor)
    label = result.cpu().detach().numpy().tolist()[0]
    label = label.index(max(label))
    out_data.append(["Final decision:", fault_type, label])
    print(out_data)
    category_list.append(label)
        

if __name__ == '__main__':
    # fault_list = Init_workflow()
    # Sample(fault_list)
    import numpy as np
    import cv2
    import random

    img=np.zeros((50,250,3),np.uint8)
    for i in range(5):
        img = cv2.rectangle(img, (i * 50, 0), ((i + 1) * 50, 50), (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)), -1)
    cv2.imwrite("vectors.png", img)
