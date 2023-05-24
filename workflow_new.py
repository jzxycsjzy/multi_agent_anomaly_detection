from __future__ import annotations
from typing import Tuple

import os
import sys
import time
import copy
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import collections
from multiprocessing import Process, Pool
import psutil

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Module
from torch.optim import Adam


# Import drain3
from drain3 import TemplateMiner
# Import some tools
from MAADModel2 import MAADModel
# from MAADModel import DecisionFusion as MAADCategory
from data_preparation import Drain_Init, RemoveSignals

# Import NLP model
from SIF.src import params, data_io, SIF_embedding

__stdout__ = sys.stdout # 标准输出就用这行
sys.stdout = open('dt.txt', 'a+')

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
loss_func = CrossEntropyLoss()
# Set training device //  if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

from matplotlib import pyplot as plt

def Init_model() -> Tuple[dict[str: MAADModel], dict[str: torch.optim.Adam]]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    optimizer_list = {}
    service_list = pd.read_csv("id_service.csv").drop(labels="Unnamed: 0", axis=1)
    for i in tqdm(range(service_list.shape[0])):
        service = service_list.loc[i]["Service"]
        model_list[service] = MAADModel()
        model_list[service].load_state_dict(torch.load("./models/Model2_00/" + service + ".pt", map_location='cpu'))
        # model_list[service].to(device)
        optimizer_list[service] = torch.optim.Adam(lr=0.0001, params=model_list[service].parameters())
    return model_list, optimizer_list

def Init_workflow() -> Tuple[list[pd.DataFrame], dict]:
    """
    Init work flow and load all data files
    """
    fault_list_1 = pd.read_csv("id_fault2.csv", index_col="id")
    fault_list_0 = pd.read_csv("id_fault.csv", index_col="id")
    return fault_list_1.to_dict()['filename']

def Sample(fault_list: dict):
    """
    Sample from each data frame and train the models
    """
    # ori_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    batch_size = 1

    models = []
    optimizers = []
    # Init Models based on batch_size
    for i in range(batch_size):
        model_dict, optim_dict = Init_model()
        models.append(model_dict)
        optimizers.append(optim_dict)
    # Init drain clusters
    tmp = Drain_Init()
    tmp.load_state("tt2")
    # print(u'加载完模型的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) )
    start_time = time.time()

    data_dir = "./data/ProcessedData/test/"
    file_list = os.listdir(data_dir)
    # Resample dataset
    file_c = {}
    for file in file_list:
        fault_type = file.split('_')[1].split('.')[0]
        if fault_type in file_c.keys():
            file_c[fault_type].append(file)
        else:
            file_c[fault_type] = [file]
    # Start training
    for epoch in range(1):
        print("Epoch {} has started.".format(epoch))
        random.shuffle(file_list)
        cur_file_list = RandomAddNormal(file_list=file_c)
        cur_file_list = file_list
        args_list = []
        pd_file_list = [[] for i in range(batch_size)]
        batch_count = 0
        for i in tqdm(range(len(cur_file_list))):
            fault_type = int(cur_file_list[i].split('_')[1].split('.')[0])
            file = os.path.join(data_dir, cur_file_list[i])
            pd_file_list[batch_count].append(file)
            if len(pd_file_list[batch_count]) == 1000:
                args = (models[batch_count], optimizers[batch_count], pd_file_list[batch_count], fault_list, tmp, fault_type)
                args_list.append(args)
                # batch_count += 1
                if len(args_list) == batch_size:
                    print("Process batch")
                    with Pool(batch_size) as p:
                        p.map(map_func, args_list)
                    # Multi_Process_Optimizing(models[batch_count], optimizers[batch_count], pd_file_list[batch_count], fault_list, tmp, fault_type, ori_memory=ori_memory)
                    batch_count = 0
                    args_list.clear()
                    for j in range(batch_size):
                        pd_file_list[j].clear()
                    # Model_Weight_Avg(models)
                    end_time = time.time()
                    print("Time cost in this batch:{}".format(end_time - start_time))
                    start_time = end_time
                    # for service in models[0].keys():
                    #     save_dir = './models/Model2_00/'
                    #     if not os.path.exists(save_dir):
                    #         os.mkdir(save_dir)
                    #     torch.save(models[0][service].state_dict(), save_dir + service + '.pt')

def Model_Weight_Avg(models_list: list):
    """
    Model average
    """
    for service in models_list[0].keys():
        worker_state_dict = []
        for i in range(len(models_list)):
            worker_state_dict.append(models_list[i][service].state_dict())
        #worker_state_dict = [x.state_dict() for x in models]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(models_list)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / len(models_list)
        #### update fed weights to fl model
        for i in range(len(models_list)):
            models_list[i][service].load_state_dict(fed_state_dict)

def RandomAddNormal(file_list: dict):
    """
    Random sample from dataset
    """
    res = []
    for label in file_list.keys():
        random.shuffle(file_list[label])
        res += file_list[label][:100]
        if label == '71':
            res += file_list[label][100:1100]
    random.shuffle(res)
    return res


def map_func(x: tuple):
    """
    Multi processing entrance.
    """
    return Multi_Process_Optimizing(x[0], x[1], x[2], x[3], x[4], x[5])


def Multi_Process_Optimizing(models: dict[str: MAADModel], optimizers: dict[str: Adam], filelist: list[str], fault_list: dict, tmp: TemplateMiner, fault_type: int, ori_memory=None):
    fault_list_0 = pd.read_csv("id_fault.csv", index_col="id").to_dict()['filename']
    fault_list_1 = pd.read_csv("id_fault2.csv", index_col="id").to_dict()['filename']
    count = 0
    time_list = []
    memory_list = []
    count_list = []
    status = True

    filesize = 0
    maadsize = 0
    for file in filelist:
        start_time = time.time()
        filesize += os.stat(file).st_size
        cur_pd = pd.read_csv(file)
        (path, file_name) = os.path.split(file)
        fault_type_0 = int(file_name.split('_')[1].split('.')[0])
        category_list = []
        feature_list = []
        fault_type_1 = GetValue(fault_list_0, fault_list_1[fault_type_0])
        # fault_type = 1 if fault_type == 71 else 0 # normal=1
        tgt = torch.tensor([fault_type_0], dtype=torch.long)
        
        # tgt = tgt.to(device)
        maadsize += WorkTrace(models=models, optimizers=optimizers, pd=cur_pd, fault_list=fault_list, tmp=tmp, fault_type=fault_type, category_list=category_list, feature_list=feature_list, tgt=tgt)
        print(maadsize, filesize)
        # Optimizing
        decision_list = []
        confidence_list = []
        for category in category_list:
            # loss = loss_func(category, tgt)
            # loss = loss / cur_pd.shape[0]
            # loss.backward(retain_graph=True)
            # Print decision
            res = category.cpu().detach().numpy().tolist()[0]
            # decision_list.append(res)
            confidence_list.append(res)
            decision_list.append(res.index(max(res)))
        for service in optimizers.keys():
            optimizers[service].step()
            optimizers[service].zero_grad()
        # print("#;{};{};{}".format(fault_type_0, decision_list, confidence_list))
        end_time = time.time()
        time_span = end_time - start_time
        time_list.append(time_span)
        memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_list.append(memory - ori_memory)
        count_list.append(count)
        count += 1
        # writeGraph(count_list, time_list, "index of trace", "time cost", "Time_curve")
        # if len(memory_list) % 10 == 0: 
        #     writeGraph(count_list[2:], memory_list[2:], "index of trace", "memory cost", "Memory usage curve")
    # print(sum(memory_list) / len(memory_list))
    # print(sum(time_list) / len(time_list))

def WorkTrace(models: dict[str: MAADModel], optimizers: dict[str: Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int, category_list: list, feature_list: list, tgt: torch.tensor):
    # dt = 0
    pd = pd.reset_index(drop=True)
    # Obtain log and span vector
    start_log = pd.loc[0]['loginfo']
    start_span = pd.loc[0]['spaninfo']
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
    # Make sure the service name
    cur_service = pd.loc[0]['service']
    cuda = next(models[cur_service].parameters()).device
    cur_tensor = None
    if cur_vectors != []:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)
    cur_tensor = cur_tensor.to(cuda) if cur_tensor != None else None
    cur_span_tensor = cur_span_tensor.to(cuda)
    info_tensor = torch.cat([cur_span_tensor, cur_tensor], dim=-2) if cur_tensor != None else cur_span_tensor
    cur_out, category = models[cur_service](info_tensor)
    cur_childs = eval(pd.loc[0]['childspans'])
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            if child_series.shape[0] != 0:
                child_service = child_series.loc[0]['service']
                # dt += sys.getsizeof(cur_out)
                # dt += WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=child_service, fault_type=fault_type, prev_out=cur_out, category_list=category_list, feature_list=feature_list, tgt=tgt)
    else:
        # It could represent that this span has come to end. Calculate the loss and backward
        category_list.append(category)
        feature_list.append(cur_out)
    #     dt += sys.getsizeof(category)
    # return dt


def WorkForward(models: dict[str: MAADModel], optimizers: dict[str:Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, cur_spans: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor, category_list: list, feature_list: list, tgt: torch.tensor):
    # dt = 0
    cuda = next(models[cur_service].parameters()).device
    
    cur_vectors = Logs2Vectors(cur_logs=cur_logs, tmp=tmp)
    cur_span_vectors = Logs2Vectors(cur_logs=[cur_spans], tmp=tmp)

    cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
    if cur_tensor != None:
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
        cur_tensor = cur_tensor.to(cuda)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)
    cur_span_tensor = cur_span_tensor.to(cuda)
    info_tensor = torch.cat([prev_out, cur_span_tensor, cur_tensor], dim=-2) if cur_tensor != None else torch.cat([prev_out, cur_span_tensor], dim=-2)
    cur_out, category = models[cur_service](info_tensor)
    cur_childs = eval(childs)
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            if child_series.shape[0] != 0:
                child_service = child_series.loc[0]['service']
                # dt += sys.getsizeof(cur_out)
                # dt += WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=child_service, fault_type=fault_type, prev_out=cur_out, category_list=category_list, feature_list=feature_list, tgt=tgt)
    # It could represent that this span has come to end. Calculate the loss and backward
    else:
        feature_list.append(cur_out)
        category_list.append(category)
    #     dt += sys.getsizeof(category)
    # return dt



def Logs2Vectors(cur_logs, tmp: TemplateMiner) -> list:
    """
    Transfer logs and spans to sentence vectors
    """
    sentences = None
    if type(cur_logs) == str:
        cur_logs = eval(cur_logs)
        sentences = [cur_logs[i][2] for i in range(len(cur_logs))]
    else:
        sentences = cur_logs
    embedding = []
    if cur_logs != []:
        for i in range(len(sentences)):
            sentence = RemoveSignals(tmp.add_log_message(sentences[i].strip())['template_mined'])
            sentences[i] = sentence
        x, m = data_io.sentences2idx(sentences=sentences, words=words)
        w = data_io.seq2weight(x, m, weight4ind)
        embedding = SIF_embedding.SIF_embedding(We, x, w, param)
    return embedding

def GetValue(d: dict, v: str):
    v = v.split("-")[0]
    for key in d.keys():
        if v == d[key]:
            return key

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    fault_list = Init_workflow()
    Sample(fault_list=fault_list)
