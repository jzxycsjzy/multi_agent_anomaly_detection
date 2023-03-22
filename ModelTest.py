from __future__ import annotations

from workflow import Init_model, Init_workflow, Logs2Vectors

import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

#Import torch
import torch

# from memory_profiler import profile

from drain3 import TemplateMiner
from MAADModel import MAADModel
from data_preparation import Drain_Init, RemoveSignals

# Import NLP model
from SIF import params, data_io, SIF_embedding

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



def Sample(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam], pds: list[pd.DataFrame], fault_list: dict):
    tmp = Drain_Init()
    tmp.load_state("tt")
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
                                WorkTrace(models=models, optimizers=optimizers, pd=cur_pd.iloc[start_index: end_index], fault_list=fault_list, tmp=tmp, fault_type=i)
                                a = cal_all(total_tensor)
                                if a != []:
                                    total_times += 1
                                    print(a)
                                    if i in a:
                                        correct += 1
                                if total_times % 100 == 0:
                                    print(total_times, correct)
                                total_tensor.clear()
                                Cal_times = random.randint(1,7)
                            count = 0
                            break
                    end_index = steps[i]
                    steps[i] += 1

def WorkTrace(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int):
    pd = pd.reset_index(drop=True)
    start_log = pd.loc[0]['loginfo']
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_service = pd.loc[0]['service']
    print(fault_list[fault_type], pd.loc[0]['traceid'])
    cur_out = None
    cur_category = None
    has_cal = False
    res = False
    if len(cur_vectors) != 0:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32)
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
        cur_out, category = models[cur_service](cur_tensor)
        has_cal = True

    cur_childs = eval(pd.loc[0]['childspans'])
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            res = WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
    if has_cal == True and res == False:
        if cur_category == None:
            return
    # if len(cur_vectors) != 0:
    #     cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32)
    #     cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
        
    # else:
    #     cur_tensor = torch.zeros([1,1,1,50], dtype=torch.float32)
    #     has_cal = True
    # cur_out, category = models[cur_service](cur_tensor)
    # has_cal = True

    # cur_childs = eval(pd.loc[0]['childspans'])
    # if len(cur_childs) != 0:
    #     for child in cur_childs:
    #         child_series = pd[pd['spanid'] == child].reset_index(drop=True)
    #         res = WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
    # if has_cal == True and res == False:
    #     if cur_category == None:
    #         return
        total_tensor.append(category)


def WorkForward(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam],  pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor) -> bool:
    cur_vectors = Logs2Vectors(cur_logs=cur_logs, tmp=tmp)
    cur_out = prev_out

    has_cal = False
    res = False
    if len(cur_vectors) != 0:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32)
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
        if prev_out != None:
            cur_tensor = torch.cat([prev_out, cur_tensor], dim=2)
        cur_out, category = models[cur_service](cur_tensor)
        has_cal = True

    cur_childs = eval(childs)
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            res = WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
    if has_cal == True and res == False:
        has_cal = True
        total_tensor.append(category)
    return has_cal

# def cal_all(tensors: list[torch.tensor]):
#     confidence_list = []
#     res = None
#     for t in tensors:
#         cache = t.detach().numpy().tolist()[0]
#         print(cache)
#         res = cache.index(max(cache))
#         confidence_list.append(res)
#         # cache = softmax(t)
#         # res = cache if res == None else cache + res
#     # if res != None:
#     #     res = res.detach().numpy().tolist()[0]
#     #     res = res.index(max(res))
#     return confidence_list
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
    test()
    # models, optimizers = Init_model()
    # pds, fault_list = Init_workflow()
    # Sample(models=models, optimizers=optimizers, pds=pds, fault_list=fault_list)
