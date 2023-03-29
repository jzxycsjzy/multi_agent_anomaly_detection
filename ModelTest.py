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
from MAADModel import MAADModel, MAADCategory
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

category_model = MAADCategory()
category_model.load_state_dict(torch.load("./models/Model2_00/category.pt"))
cur_seq = []
cur_faults = []

def Init_model() -> dict[str: MAADModel]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    service_list = pd.read_csv("id_service.csv").drop(labels="Unnamed: 0", axis=1)
    save_dir = './models/Model2_00/'
    for i in tqdm(range(service_list.shape[0])):
        service = service_list.loc[i]["Service"]
        model_list[service] = MAADModel()
        model_list[service].eval()
        model_list[service].load_state_dict(torch.load(save_dir + service + ".pt"))
    return model_list

def Init_workflow() -> Tuple[list[pd.DataFrame], dict]:
    fault_list = pd.read_csv("id_fault.csv", index_col="id")
    src_path = 'data/ProcessedData/test'
    pds = []
    for i in tqdm(range(fault_list.shape[0])):
        f = os.path.join(src_path, "test_" + fault_list.loc[i]['filename'])
        if not os.path.exists(f):
            print("file not exists!")
            continue
        dt = pd.read_csv(f)
        pds.append(dt)
    return pds, fault_list.to_dict()['filename']

def Sample(models: dict[str: MAADModel], pds: list[pd.DataFrame], fault_list: dict):
    tmp = Drain_Init()
    tmp.load_state("tt2")
    steps = [0] * 15
    traces = 0

    total_times = 0
    correct = 0

    Cal_times = random.randint(1,7)
    for epoch in range(15):
        while True:
            status = True
            for i in range(15):
                if pds[i].shape[0] > steps[i]:
                    status = False
                    
                    break
            if status:
                steps = [0] * 15
                break
            for i in range(15):
                cur_pd = pds[i]
                count = 0
                start_index = steps[i]
                end_index = steps[i]
                while steps[i] < cur_pd.shape[0]:
                    if cur_pd.loc[steps[i]]['parentspan'] == '-1' or cur_pd.loc[steps[i]]['parentspan'] == -1 or steps[i] == cur_pd.shape[0]:
                        count += 1
                        if count == 2:
                            end_index = steps[i]
                            traces += 1
                            if traces % Cal_times == 0:
                                WorkTrace(models=models, pd=cur_pd.iloc[start_index: end_index], fault_list=fault_list, tmp=tmp, fault_type=i)
                                features = torch.cat(cur_seq, dim=1)
                                features = features.unsqueeze(0)
                                category = category_model(features)
                                category = category.to("cpu")
                                category = category.detach().numpy().tolist()[0][0][0]
                                total_times += 1
                                print(category)
                                if category.index(max(category)) == i:
                                    correct += 1
                                if total_times % 100 == 0:
                                    print(total_times, correct)
                                total_tensor.clear()
                                Cal_times = random.randint(1,7)
                                cur_seq.clear()
                            count = 0
                            break
                    end_index = steps[i]
                    steps[i] += 1

def WorkTrace(models: dict[str: MAADModel], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int):
    pd = pd.reset_index(drop=True)
    # Obtain log and span vector
    start_log = pd.loc[0]['loginfo']
    start_span = pd.loc[0]['spaninfo']
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
    # Make sure the service name
    cur_service = pd.loc[0]['service']
    print(fault_list[fault_type], pd.loc[0]['traceid'])
    cur_tensor = None
    if cur_vectors != []:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
        cur_tensor = cur_tensor.unsqueeze(0)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0)

    cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)
    cur_childs = eval(pd.loc[0]['childspans'])
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            WorkForward(models=models, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
    else:
        cur_seq.append(cur_out)
        cur_faults.append(category)


def WorkForward(models: dict[str: MAADModel],  pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, cur_spans: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor):
    cur_vectors = Logs2Vectors(cur_logs=cur_logs, tmp=tmp)
    cur_span_vectors = Logs2Vectors(cur_logs=[cur_spans], tmp=tmp)
    cur_out = prev_out

    cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
    if cur_tensor != None:
        cur_tensor = cur_tensor.unsqueeze(0)
        cur_tensor = torch.cat([prev_out, cur_tensor], dim=1)
    else:
        cur_tensor = prev_out
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0)
    cur_span_vectors = torch.cat([prev_out, cur_span_tensor], dim=1)
    cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)

    cur_childs = eval(childs)
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            WorkForward(models=models, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
    # It could represent that this span has come to end. Calculate the loss and backward
    else:
        cur_seq.append(cur_out)
        cur_faults.append(category)

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

def test():
    a = [1,4,7,2,3,0]
    b = a.copy()
    a.sort(reverse=True)
    r = 0
    if len(a) > 3:
        r = 3
    else:
        r = len(a)
    f2 = [b.index(a[i]) for i in range(r)]
    print(f2)

if __name__ == '__main__':
    models = Init_model()
    pds, fault_list = Init_workflow()
    Sample(models=models, pds=pds, fault_list=fault_list)
