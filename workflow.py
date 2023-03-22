from __future__ import annotations
from typing import Tuple

import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

#Import torch
import torch

# from memory_profiler import profile

from drain3 import TemplateMiner
from MAADModel import MAADModel
from data_preparation import Drain_Init, RemoveSignals

# Import NLP model
from SIF import params, data_io, SIF_embedding

__stdout__ = sys.stdout # 标准输出就用这行
open("log.txt", 'w+')
sys.stdout = open('log.txt', 'a')
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

# Set loss function
loss_func = torch.nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Init_model() -> Tuple[dict[str: MAADModel], dict[str: torch.optim.Adam]]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    optimizer_list = {}
    service_list = pd.read_csv("id_service.csv").drop(labels="Unnamed: 0", axis=1)
    # service_list = service_list.sample(frac=1).reset_index(drop=True)
    # print(service_list.loc[0])
    for i in tqdm(range(service_list.shape[0])):
        service = service_list.loc[i]["Service"]
        model_list[service] = MAADModel()
        # model_list[service].to(device)
        # model_list[service].load_state_dict(torch.load("./models/Model_02_lr1e-4/" + service + ".pt"))
        optimizer_list[service] = torch.optim.Adam(lr=0.001, params=model_list[service].parameters())
        optimizer_list[service].zero_grad()
    return model_list, optimizer_list

def Init_workflow() -> Tuple[list[pd.DataFrame], dict]:
    fault_list = pd.read_csv("id_fault.csv", index_col="id")
    src_path = 'data/ProcessedData/'
    pds = []
    for i in tqdm(range(fault_list.shape[0])):
        f = os.path.join(src_path, fault_list.loc[i]['filename'])
        if not os.path.exists(f):
            continue
        dt = pd.read_csv(f)
        pds.append(dt)
    return pds, fault_list.to_dict()['filename']

def Sample(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam], pds: list[pd.DataFrame], fault_list: dict):
    tmp = Drain_Init()
    tmp.load_state("tt")
    steps = [0] * 15
    traces = 0
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
                            WorkTrace(models=models, optimizers=optimizers, pd=cur_pd.iloc[start_index: end_index], fault_list=fault_list, tmp=tmp, fault_type=i)
                            for service in optimizers.keys():
                                optimizers[service].zero_grad()
                            count = 0
                            if traces % 100 == 0:
                                for service in models.keys():
                                    torch.save(models[service].state_dict(), './models/Model03/' + service + '.pt')
                            break
                    end_index = steps[i]
                    steps[i] += 1

# def WorkTrace(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int):
#     pd = pd.reset_index(drop=True)
#     start_log = pd.loc[0]['loginfo']
#     cur_vectors = Logs2Vectors(start_log, tmp=tmp)
#     cur_service = pd.loc[0]['service']
#     print(fault_list[fault_type], pd.loc[0]['traceid'])
#     cur_out = None
#     cur_category = None
#     has_cal = False
#     res = False
#     if len(cur_vectors) != 0:
#         cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32)
#         cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
#     else:
#         cur_tensor = torch.zeros([1,1,1,50], dtype=torch.float32)
#         has_cal = True
#     #cuda = next(models[cur_service].parameters()).device
#     #cur_tensor = cur_tensor.to(cuda)
#     cur_out, category = models[cur_service](cur_tensor)
#     has_cal = True

#     cur_childs = eval(pd.loc[0]['childspans'])
#     if len(cur_childs) != 0:
#         for child in cur_childs:
#             child_series = pd[pd['spanid'] == child].reset_index(drop=True)
#             res = WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
#     if has_cal == True and res == False:
#         if cur_category == None:
#             return
#         # It could represent that this span has come to end. Calculate the loss and backward
#         tgt = torch.tensor([fault_type], dtype=torch.long)
#         #tgt = tgt.to(cuda)
#         loss = loss_func(category, tgt)
#         loss.backward(retain_graph=True)
#         for service in optimizers.keys():
#             optimizers[service].step()
#             optimizers[service].zero_grad()
#         print(loss)


# def WorkForward(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam],  pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor) -> bool:
#     cur_vectors = Logs2Vectors(cur_logs=cur_logs, tmp=tmp)
#     cur_out = prev_out

#     has_cal = False
#     res = False
#     if len(cur_vectors) != 0:
#         cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32)
#         cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
#         cur_tensor = torch.cat([prev_out, cur_tensor], dim=2)
#     else:
#         cache = torch.zeros([1,1,1,50], dtype=torch.float32)
#         #cache = cache.to(device)
#         cur_tensor = torch.cat([cur_out, cache], dim=2)
#     #cuda = next(models[cur_service].parameters()).device
#     #cur_tensor = cur_tensor.to(cuda)
#     cur_out, category = models[cur_service](cur_tensor)
#     has_cal = True

#     cur_childs = eval(childs)
#     if len(cur_childs) != 0:
#         for child in cur_childs:
#             child_series = pd[pd['spanid'] == child].reset_index(drop=True)
#             res = WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out)
#     if has_cal == True and res == False:
#         # It could represent that this span has come to end. Calculate the loss and backward
#         tgt = torch.tensor([fault_type], dtype=torch.long)
#         #tgt = tgt.to(cuda)
#         loss = loss_func(category, tgt)
#         loss.backward(retain_graph=True)
#         for service in optimizers.keys():
#             optimizers[service].step()
#             optimizers[service].zero_grad()
#         print(loss)
#         has_cal = True

#     return has_cal

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
        # It could represent that this span has come to end. Calculate the loss and backward
        tgt = torch.tensor([fault_type], dtype=torch.long)
        loss = loss_func(category, tgt)
        loss.backward(retain_graph=True)
        for service in optimizers.keys():
            optimizers[service].step()
            # optimizers[service].zero_grad()
        print(loss)


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
        # It could represent that this span has come to end. Calculate the loss and backward
        tgt = torch.tensor([fault_type], dtype=torch.long)
        loss = loss_func(category, tgt)
        loss.backward(retain_graph=True)
        for service in optimizers.keys():
            optimizers[service].step()
            # optimizers[service].zero_grad()
        print(loss)
        has_cal = True

    return has_cal



def Logs2Vectors(cur_logs: list, tmp: TemplateMiner) -> list:
    cur_logs = eval(cur_logs)
    embedding = []
    if cur_logs != []:
        sentenses = [cur_logs[i][2] for i in range(len(cur_logs))]
        for i in range(len(sentenses)):
            sentence = RemoveSignals(tmp.add_log_message(sentenses[i].strip())['template_mined'])
            sentenses[i] = sentence
        x, m = data_io.sentences2idx(sentences=sentenses, words=words)
        w = data_io.seq2weight(x, m, weight4ind)
        embedding = SIF_embedding.SIF_embedding(We, x, w, param)
    return embedding

            

        

if __name__ == '__main__':
    models, optimizers = Init_model()
    pds, fault_list = Init_workflow()
    Sample(models=models, optimizers=optimizers, pds=pds, fault_list=fault_list)
