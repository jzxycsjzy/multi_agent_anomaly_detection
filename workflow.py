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

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Module
from torch.optim import Adam


# Import drain3
from drain3 import TemplateMiner
# Import some tools
from MAADModel import MAADModel
from MAADModel import DecisionFusion as MAADCategory
from data_preparation import Drain_Init, RemoveSignals

# Import NLP model
from SIF.src import params, data_io, SIF_embedding

__stdout__ = sys.stdout # 标准输出就用这行
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
loss_func = CrossEntropyLoss()
# Set training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Save current output of model


def Init_model() -> Tuple[dict[str: MAADModel], dict[str: torch.optim.Adam], MAADCategory, Adam]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    optimizer_list = {}
    service_list = pd.read_csv("id_service.csv").drop(labels="Unnamed: 0", axis=1)
    # service_list = service_list.sample(frac=1).reset_index(drop=True)
    # print(service_list.loc[0])
    category_model = MAADCategory()
    category_model.to(device)
    category_optimizer = Adam(lr=0.003, params=category_model.parameters())
    for i in tqdm(range(service_list.shape[0])):
        service = service_list.loc[i]["Service"]
        # service = "unified"
        model_list[service] = MAADModel()
        
        # model_list[service].load_state_dict(torch.load("/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/models/Model2_00/" + service + ".pt"))
        model_list[service].to(device)
        optimizer_list[service] = torch.optim.Adam(lr=0.0001, params=model_list[service].parameters())
        # optimizer_list[service].zero_grad()
        # break
    return model_list, optimizer_list, category_model, category_optimizer

def Init_workflow() -> Tuple[list[pd.DataFrame], dict]:
    """
    Init work flow and load all data files
    """
    fault_list = pd.read_csv("id_fault2.csv", index_col="id")
    # src_path = 'data/ProcessedData/train'
    # pds = []
    # for i in tqdm(range(fault_list.shape[0])):
    #     f = os.path.join(src_path, "train_" + fault_list.loc[i]['filename'])
    #     if not os.path.exists(f):
    #         print("file not exists!")
    #         continue
    #     dt = pd.read_csv(f)
    #     pds.append(dt)
    return fault_list.to_dict()['filename']

# def Sample(models: dict[str: MAADModel], optimizers: dict[str: torch.optim.Adam], pds: list[pd.DataFrame], fault_list: dict):
#     """
#     Sample from each data frame and train the models
#     """
#     tmp = Drain_Init()
#     tmp.load_state("tt2")
#     steps = [0] * 15
#     traces = 1

#     batch_size = 1
#     batch_count = 0
    
#     start_time = time.time()
#     for epoch in range(15):
#         print("Epoch {} has started.".format(epoch))
#         while True:
#             status = True
#             for i in range(15):
#                 if pds[i].shape[0] > steps[i]:
#                     status = False
#                     break
#             if status:
#                 steps = [0] * 15
#                 break
#             for i in range(15):
#                 cur_pd = pds[i]
#                 count = 0
#                 start_index = steps[i]
#                 end_index = steps[i]
#                 while steps[i] < cur_pd.shape[0]:
                    
#                     if cur_pd.loc[steps[i]]['parentspan'] == '-1' or cur_pd.loc[steps[i]]['parentspan'] == -1 or steps[i] == cur_pd.shape[0]:
#                         count += 1
#                         if count == 2:
#                             end_index = steps[i]
#                             traces += 1
#                             WorkTrace(models=models, optimizers=optimizers, pd=cur_pd.iloc[start_index: end_index], fault_list=fault_list, tmp=tmp, fault_type=i)
#                             # Backward
#                             tgt = torch.tensor([i], dtype=torch.long)
#                             tgt = tgt.to(device)
#                             for category in cur_faults:
#                                 loss = loss_func(category, tgt)
#                                 if loss < torch.tensor(0.1, dtype=torch.float):
#                                     break
#                                 loss.backward(retain_graph=True)
#                                 res = category.cpu().detach().numpy().tolist()[0]
#                                 res = res.index(max(res))
#                                 print(res, loss)
#                             if len(cur_faults) <= 5:
#                                 for i in range(6):
#                                     for service in optimizers.keys():
#                                         optimizers[service].step()
#                                 for service in optimizers.keys():
#                                     optimizers[service].zero_grad()

#                             # features = torch.cat(cur_seq, dim=1)
#                             # features = features.unsqueeze(0)
#                             # category = category_model(features)
#                             # category = category.squeeze(0).squeeze(0)
#                             # tgt = torch.tensor([i], dtype=torch.long)
#                             # tgt = tgt.to(device)
#                             # loss = loss_func(category, tgt)
#                             # threshold = torch.tensor(1.817, dtype=torch.float32)
#                             # loss = loss - threshold
#                             # loss.backward(retain_graph=False)
#                             # print(loss)
#                             # print(category)
                            
#                             # for service in optimizers.keys():
#                             #         optimizers[service].step()
#                             #         optimizers[service].zero_grad()
#                             # batch_count += 1
#                             # if batch_count % 100 == 0:
#                             #     end_time = time.time()
#                             #     print("Time cost in this 100 traces:{}".format(end_time - start_time))
#                             #     start_time = end_time
#                             # batch_loss.append(loss)
#                             batch_count += 1
#                             if batch_count % batch_size == 0:
#                                 print("Optimized.")
#                                 end_time = time.time()
#                                 print("Time cose in this batch:{}".format(end_time - start_time))
#                                 start_time = end_time
#                             # cur_seq.clear()
#                             cur_faults.clear()
#                             count = 0
#                             if traces % 1000 == 0:
#                                 for service in models.keys():
#                                     save_dir = './models/Model2_00/'
#                                     if not os.path.exists(save_dir):
#                                         os.mkdir(save_dir)
#                                     torch.save(models[service].state_dict(), save_dir + service + '.pt')
#                                     # torch.save(category_model.state_dict(), save_dir + "category.pt")
#                             break
#                     end_index = steps[i]
#                     steps[i] += 1

def Sample(fault_list: dict):
    """
    Sample from each data frame and train the models
    """
    batch_size = 8

    category_models = []
    category_optimizers = []
    models = []
    optimizers = []
    for i in range(batch_size):
        model_dict, optim_dict, category, category_optim = Init_model()
        models.append(model_dict)
        optimizers.append(optim_dict)
        category_models.append(category)
        category_optimizers.append(category_optim)

    tmp = Drain_Init()
    tmp.load_state("tt2")
    
    start_time = time.time()

    data_dir = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/ProcessedData/train/"
    file_list = os.listdir(data_dir)
    # Resample
    file_c = {}
    for file in file_list:
        fault_type = file.split('_')[1].split('.')[0]
        if fault_type in file_c.keys():
            file_c[fault_type].append(file)
        else:
            file_c[fault_type] = [file]
    # for t in file_c.keys():
    #     print(len(file_c[t]))
    # exit(0)
    for epoch in range(45):
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
            # cur_pd = pd.read_csv(file)
            pd_file_list[batch_count].append(file)
            if len(pd_file_list[batch_count]) == 300:
                args = (models[batch_count], optimizers[batch_count], category_models[batch_count], category_optimizers[batch_count], pd_file_list[batch_count], fault_list, tmp, fault_type)
                args_list.append(args)
                batch_count += 1
                if len(args_list) == batch_size:
                    print("Process batch")
                    with Pool(batch_size) as p:
                        p.map(map_func, args_list)
                    batch_count = 0
                    args_list.clear()
                    for j in range(batch_size):
                        pd_file_list[j].clear()
                    # WorkTrace(models=models, optimizers=optimizers, pd=cur_pd, fault_list=fault_list, tmp=tmp, fault_type=fault_type)
                    # is_normal = 1 if fault_type == 71 else 0
                    # tgt = torch.tensor([is_normal], dtype=torch.float32)
                    # tgt = tgt.to(device)
                    # for category in cur_faults:
                    #     loss = loss_func(category, tgt)
                    #     # threshold = torch.tensor(0.1, dtype=torch.float)
                    #     # threshold = threshold.to(device)
                    #     # if loss < threshold:
                    #     #     break
                    #     loss.backward(retain_graph=True)
                    #     res = category.cpu().detach().numpy().tolist()[0]
                    #     # res = res.index(max(res))
                    #     print(res, loss)
                    # recursion_steps = 1
                    # if len(cur_faults) <= 5:
                    #     recursion_steps = 6
                    # for i in range(recursion_steps):
                    #     for service in optimizers.keys():
                    #         optimizers[service].step()
                    # for service in optimizers.keys():
                    #     optimizers[service].zero_grad()
                    # print("Optimized.")
                    Model_Weight_Avg(models)
                    Category_Weight_Avg(category_models)
                    end_time = time.time()
                    print("Time cost in this batch:{}".format(end_time - start_time))
                    start_time = end_time
                    for service in models[0].keys():
                        save_dir = './models/Model2_00/'
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        torch.save(models[0][service].state_dict(), save_dir + service + '.pt')
                        torch.save(category_models[0].state_dict(), save_dir + "category.pt")

def Model_Weight_Avg(models_list: list):
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

def Category_Weight_Avg(models_list: list):
    worker_state_dict = []
    for i in range(len(models_list)):
        worker_state_dict.append(models_list[i].state_dict())
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
        models_list[i].load_state_dict(fed_state_dict)

def RandomAddNormal(file_list: dict):
    res = []
    for label in file_list.keys():
        random.shuffle(file_list[label])
        res += file_list[label][:100]
    random.shuffle(res)
    return res



def map_func(x: tuple):
    return Multi_Process_Optimizing(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])


def Multi_Process_Optimizing(models: dict[str: MAADModel], optimizers: dict[str: Adam], category_model: MAADCategory, category_optimizer: Adam, filelist: list[str], fault_list: dict, tmp: TemplateMiner, fault_type: int):
    for file in filelist:
        cur_pd = pd.read_csv(file)
        (path, file_name) = os.path.split(file)
        fault_type = int(file_name.split('_')[1].split('.')[0])
        category_list = []
        feature_list = []
        fault_type = 1 if fault_type == 71 else 0 # normal=1
        tgt = torch.tensor([fault_type], dtype=torch.long)
        tgt = tgt.to(device)
        WorkTraceNew(models=models, optimizers=optimizers, pd=cur_pd, fault_list=fault_list, tmp=tmp, fault_type=fault_type, category_list=category_list, feature_list=feature_list, tgt=tgt, category_model=category_model, category_optimizer=category_optimizer)
        # is_normal = 1 if fault_type == 71 else 0
        # Result get
        # feature = torch.cat(feature_list, dim=-2)
        # out = category_model(feature)
        
        # # threshold = torch.tensor(1.0, dtype=torch.float32)
        # # loss = loss_func(out, tgt)
        # # loss.backward(retain_graph=True)
        # # res = out.cpu().detach().numpy().tolist()[0]
        # # print(file, res.index(max(res)), fault_type)
        # for category in category_list:
        #     loss = loss_func(category, tgt)
        #     # loss = loss - threshold
        #     loss.backward(retain_graph=True)
        #     res = category.cpu().detach().numpy().tolist()[0]
        #     res = res.index(max(res))
        #     print(file, res, loss)
        # recursion_steps = 1
        # if len(category_list) <= 5:
        #     recursion_steps = 6
        # for i in range(recursion_steps):
        #     for service in optimizers.keys():
        #         optimizers[service].step()
        #     category_optimizer.step()
        # for service in optimizers.keys():
        #     optimizers[service].zero_grad()
        
        # category_optimizer.zero_grad()
        # print("Optimized.")

def WorkTrace(models: dict[str: MAADModel], optimizers: dict[str: Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int, category_list: list, feature_list: list, tgt: torch.tensor):
    pd = pd.reset_index(drop=True)
    # Obtain log and span vector
    start_log = pd.loc[0]['loginfo']
    start_span = pd.loc[0]['spaninfo']
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
    # Make sure the service name
    cur_service = pd.loc[0]['service']
    cur_service = "unified"
    cuda = next(models[cur_service].parameters()).device
    # print(fault_type, fault_list[fault_type], pd.loc[0]['traceid'])
    cur_tensor = None
    if cur_vectors != []:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)

    cur_tensor = cur_tensor.to(cuda) if cur_tensor != None else None
    cur_span_tensor = cur_span_tensor.to(cuda)
    cur_out, category = models[cur_service](cur_tensor, cur_span_tensor)
    loss = loss_func(category, tgt)
    loss.backward()
    res = category.cpu().detach().numpy().tolist()[0]
    print(fault_type, res.index(max(res)))
    category_list.append(category)
    optimizers[cur_service].step()
    optimizers[cur_service].zero_grad()
    cur_out = cur_out.detach()
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
                WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out, category_list=category_list, feature_list=feature_list, tgt=tgt)
    else:
        # It could represent that this span has come to end. Calculate the loss and backward
        feature_list.append(cur_out)
        
        # tgt = torch.tensor([fault_type], dtype=torch.long)
        # loss = loss_func(category, tgt)
        # loss.backward(retain_graph=True)
        # for service in optimizers.keys():
        #     optimizers[service].step()
        #     # optimizers[service].zero_grad()
        # print(loss)


def WorkForward(models: dict[str: MAADModel], optimizers: dict[str:Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, cur_logs: list, cur_spans: list, childs: list, cur_service: str, fault_type: int, prev_out: torch.tensor, category_list: list, feature_list: list, tgt: torch.tensor):
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
    loss = loss_func(category, tgt)
    loss.backward()
    category_list.append(category)
    optimizers[cur_service].step()
    optimizers[cur_service].zero_grad()
    cur_out = cur_out.detach()
    res = category.cpu().detach().numpy().tolist()[0]
    print(fault_type, res.index(max(res)))
    cur_childs = eval(childs)
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_series = pd[pd['spanid'] == child].reset_index(drop=True)
            if child_series.shape[0] != 0:
                WorkForward(models=models, optimizers=optimizers, pd=pd, fault_list=fault_list, tmp=tmp, cur_logs=child_series.loc[0]['loginfo'], cur_spans=child_series.loc[0]['spaninfo'], childs=child_series.loc[0]['childspans'], cur_service=cur_service, fault_type=fault_type, prev_out=cur_out, category_list=category_list, feature_list=feature_list, tgt=tgt)
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

def WorkTraceNew(models: dict[str: MAADModel], optimizers: dict[str: Adam], pd: pd.DataFrame, fault_list: dict, tmp: TemplateMiner, fault_type: int, category_list: list, feature_list: list, tgt: torch.tensor, category_model: MAADCategory, category_optimizer: Adam):
    pd = pd.reset_index(drop=True)
    span_data_collection = {}
    log_data_collection = {}
    # cuda = next(models["ts-order-service"].parameters()).device
    cuda = next(models["ts-order-service"].parameters()).device
    for span_idx in range(pd.shape[0]):
        dt = pd.loc[span_idx]
        service = dt['service']
        # service = "unified"
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
    feature_list = []
    for service in span_data_collection.keys():
        # if service not in span_data_collection.keys():
        #     out = torch.zeros([1, 72], dtype=torch.float32)
        #     out = out.to(cuda)
        #     res_list.append(out)
        #     features = torch.zeros([1, 300], dtype=torch.float32)
        #     features = features.to(cuda)
        #     feature_list.append(features)
        #     continue
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
        # feature_list.append(features)
        res = category.cpu().detach().numpy().tolist()[0]
        res_list.append(res)
        res = res.index(max(res))
        loss = loss_func(category, tgt)
        loss.backward()
        out_data.append([fault_type, res])
        optimizers[service].step()
        optimizers[service].zero_grad()
    # category_tensor = torch.cat(res_list, dim=-1)
    # feature_tensor = torch.cat(feature_list, dim=-1)
    # result = category_model(category_tensor, feature_tensor)
    # loss = loss_func(result, tgt)
    # loss.backward()
    # for service in span_data_collection.keys():
    #     optimizers[service].step()
    #     optimizers[service].zero_grad()
    # category_optimizer.step()
    # category_optimizer.zero_grad()
    # label = result.cpu().detach().numpy().tolist()[0]
    # label = label.index(max(label))
    # out_data.append(["Final decision:", fault_type, label, loss])
    r = 0
    for res in res_list:
        if res.index(max(res)) == 1:
            r = 1
        # for i in range(2):
        #     if res[i] == 1:
        #         r = 1
    print(out_data, r)
        

            
if __name__ == '__main__':
    pass
