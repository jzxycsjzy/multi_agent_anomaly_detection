from __future__ import annotations
from typing import Tuple

import os
import sys
import random
from tqdm import tqdm
import pandas as pd
import collections
from multiprocessing import Pool
import argparse

# Import torch modules
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


# Import drain3
from drain3 import TemplateMiner
# Import model class
from model.MAADModel import MAADModel
# Import some tools
from data_preparation import Drain_Init, RemoveSignals

# Import sentence vectorization module
from SIF.src import params, data_io, SIF_embedding
# Save the output to the file
# __stdout__ = sys.stdout
# sys.stdout = open('dt.txt', 'a+')

# Init SIF parameters
wordfile = "./GloVeModel/vectors.txt" # word vector file, can be downloaded from GloVe website
weightfile = "./GloVeModel/vocab.txt" # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
# load word vectors
words = None
We = None
# load word weights
word2weight = None # word2weight['str'] is the weight for the word 'str'
weight4ind = None # weight4ind[i] is the weight for the i-th word
# set parameters
param = params.params()
param.rmpc = rmpc

# Set loss function
loss_func = CrossEntropyLoss()

# Set Pre config
service_pos = 2
span_pos = 4
error_pos = 9
event_pos = 10
child_pos = 11
log_pos = 12

def Init_model(servicelist: str, error_types: int, isTrain: bool) -> Tuple[dict[str: MAADModel], dict[str: torch.optim.Adam]]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    optimizer_list = {}
    service_list = pd.read_csv(servicelist)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in tqdm(range(service_list.shape[0])):
        service = service_list.loc[i]["Service"]
        model_list[service] = MAADModel(error_types)
        if not isTrain:
            model_list[service].load_state_dict(torch.load("./models/" + service + ".pt"))
        # model_list[service].to(device)
        optimizer_list[service] = torch.optim.Adam(lr=0.0001, params=model_list[service].parameters())
    return model_list, optimizer_list

def Init_workflow(fault_list: str) -> Tuple[list[pd.DataFrame], dict]:
    """
    Init fualt list
    """
    fault_list = pd.read_csv(fault_list)
    fault_list = fault_list.to_dict()['faultname']
    fault_list = {v : k for k, v in fault_list.items()}
    return fault_list

def Sample(fault_list: dict, arguments):
    """
    Sample from each data frame and train the models
    """
    # ori_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    batch_size = arguments.batch

    models = []
    optimizers = []
    # Init Models based on batch_size
    for i in range(batch_size):
        model_dict, optim_dict = Init_model(arguments.servicelist, arguments.errortypes, arguments.train)
        models.append(model_dict)
        optimizers.append(optim_dict)
    # Init drain clusters
    tmp = Drain_Init()
    tmp.load_state(arguments.drain)

    # data_dir = "./data/ProcessedData/test/"
    file_list = os.listdir(arguments.trainset)
    # Resample dataset
    if arguments.labelmode == 0:
        file_c = {}
        for file in file_list:
            fault_type = file.split('_')[1].split('.')[0]
            if fault_type in file_c.keys():
                file_c[fault_type].append(file)
            else:
                file_c[fault_type] = [file]
    else:
        file_c = file_list
    args_list = []
    pd_file_list = [[] for i in range(batch_size)]
    batch_count = 0
    # Start training
    for epoch in range(5):
        print("Epoch {} has started.".format(epoch))
        random.shuffle(file_list)
        # cur_file_list = RandomAddNormal(file_list=file_c)
        cur_file_list = file_list
        for i in tqdm(range(len(cur_file_list))):
            # fault_type = int(cur_file_list[i].split('_')[1].split('.')[0])
            file = os.path.join(arguments.trainset, cur_file_list[i])
            pd_file_list[batch_count].append(file)
            if len(pd_file_list[batch_count]) == 100000:
                args = (models[batch_count], optimizers[batch_count], pd_file_list[batch_count], fault_list, tmp, arguments)
                args_list.append(args)
                batch_count += 1
                if len(args_list) == batch_size:
                    print("Process batch")
                    with Pool(batch_size) as p:
                        p.map(map_func, args_list)
                    # Multi_Process_Optimizing(models[batch_count], optimizers[batch_count], pd_file_list[batch_count], fault_list, tmp, fault_type, ori_memory=ori_memory)
                    batch_count = 0
                    args_list.clear()
                    for j in range(batch_size):
                        pd_file_list[j].clear()
                    Model_Weight_Avg(models)
                    for service in models[0].keys():
                        save_dir = './models/'
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        torch.save(models[0][service].state_dict(), save_dir + service + '.pt')

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


def Multi_Process_Optimizing(models: dict[str: MAADModel], optimizers: dict[str: Adam], filelist: list[str], fault_list: dict, tmp: TemplateMiner, arguments):
    for file in filelist:
        # cur_pd = pd.read_csv(file)
        # cur_pd.to_csv('test.csv')
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
        _, file_name = os.path.split(file)
        fault = 0
        if arguments.labelmode == 0:
            fault = int(file_name.split('_')[1])
        category_list = []
        feature_list = []
        error_list = []
        WorkTrace(models=models, optimizers=optimizers, lines=lines, fault_list=fault_list, tmp=tmp, error_list=error_list, category_list=category_list, feature_list=feature_list, label_mode=arguments.labelmode)
        # Optimizing phase
        decision_list = []
        tgt = torch.tensor([fault], dtype=torch.long)
        for i in range(len(category_list)):
            category = category_list[i]
            res = category.cpu().detach().numpy().tolist()[0]
            if arguments.labelmode == 0:
                fault = error_list[i]
                tgt = torch.tensor([fault], dtype=torch.long)
            if arguments.train:
                loss = loss_func(category, tgt)
                loss = loss / len(lines)
                loss.backward(retain_graph=True)
            # decision_list.append(res)
            # confidence_list.append(res)
            decision_list.append(res.index(max(res)))
        out_str = ""
        if arguments.labelmode == 0:
            outstr = [decision_list, error_list]
        else:
            outstr = [decision_list, fault]
        outfile_name = "MAADout_train.txt" if arguments.train == True else "MAADout_test.txt"
        with open(outfile_name, 'a+') as f:
            f.write(outstr)
            f.write("\n")
        if arguments.train:
            for service in optimizers.keys():
                optimizers[service].step()
                optimizers[service].zero_grad()
        # print("#;{};{};{}".format(fault_type_0, decision_list, confidence_list))

def WorkTrace(models: dict[str: MAADModel], optimizers: dict[str: Adam], lines: list, fault_list: dict, tmp: TemplateMiner, error_list: list, category_list: list, feature_list: list, label_mode: int):
    # Obtain log and span vector
    start_trace_info = lines[0].split(',')
    
    # Init data
    cur_service = start_trace_info[service_pos]
    start_span = start_trace_info[event_pos]
    cur_childs = [] if str(start_trace_info[child_pos])=="0" else start_trace_info[child_pos].rstrip().split(';')
    # The first span should have at least one child span, otherwise this data is error.
    if cur_childs == []:
        return
    start_log = [] if len(start_trace_info) <= log_pos else ','.join(start_trace_info[log_pos:]).split(';')
    
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
    # Make sure the service name
    # cuda = next(models[cur_service].parameters()).device
    cur_tensor = None
    if cur_vectors != []:
        cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)
    # cur_tensor = cur_tensor.to(cuda) if cur_tensor != None else None
    # cur_span_tensor = cur_span_tensor.to(cuda)
    info_tensor = torch.cat([cur_span_tensor, cur_tensor], dim=-2) if cur_tensor != None else cur_span_tensor
    cur_out, category = models[cur_service](info_tensor)
    if label_mode == 1:
        error_type = fault_list[int(start_trace_info[error_pos])]
        error_list.append(error_type)
        category_list.append(category)
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_line = ""
            for i in range(1, len(lines)):
                cur_line = lines[i]
                cur_info = cur_line.split(',')[span_pos]
                if cur_info == child:
                    child_line = cur_line
                    break
            if child_line != "":
                WorkForward(lines=lines, models=models, optimizers=optimizers, line=child_line, fault_list=fault_list, tmp=tmp, error_list=error_list, prev_out=cur_out, category_list=category_list, feature_list=feature_list, label_mode=label_mode)
    else:
        # It could represent that this span has come to end. Calculate the loss and backward
        if label_mode == 0:
            category_list.append(category)


def WorkForward(lines: list, models: dict[str: MAADModel], optimizers: dict[str:Adam], line: str, fault_list: dict, tmp: TemplateMiner, error_list: list, prev_out: torch.tensor, category_list: list, feature_list: list, label_mode: int):
    # cuda = next(models[cur_service].parameters()).device
    start_trace_info = line.split(',')
    # Init data
    # prev_out = prev_out
    # zeros = torch.tensor([[0] * 296], dtype=torch.float32)
    # prev_out = torch.cat([prev_out, zeros], dim=-1).unsqueeze(0).unsqueeze(0)
    
    cur_service = start_trace_info[service_pos]
    start_span = start_trace_info[event_pos]
    
    cur_childs = [] if str(start_trace_info[child_pos])=="0" else start_trace_info[child_pos].rstrip().split(';')
    # The first span should have at least one child span, otherwise this data is error.
    if cur_childs == []:
        return
    start_log = [] if len(start_trace_info) <= log_pos else ','.join(start_trace_info[log_pos:]).split(';')
    
    cur_vectors = Logs2Vectors(start_log, tmp=tmp)
    cur_span_vectors = Logs2Vectors([start_span], tmp=tmp)
    if cur_span_vectors == []:
        return
    cur_tensor = torch.tensor(cur_vectors, dtype=torch.float32) if cur_vectors != [] else None
    if cur_tensor != None:
        cur_tensor = cur_tensor.unsqueeze(0).unsqueeze(0)
        # cur_tensor = cur_tensor.to(cuda)
    cur_span_tensor = torch.tensor(cur_span_vectors, dtype=torch.float32)
    cur_span_tensor = cur_span_tensor.unsqueeze(0).unsqueeze(0)
    # cur_span_tensor = cur_span_tensor.to(cuda)
    info_tensor = torch.cat([prev_out, cur_span_tensor, cur_tensor], dim=-2) if cur_tensor != None else torch.cat([prev_out, cur_span_tensor], dim=-2)
    cur_out, category = models[cur_service](info_tensor)
    
    if label_mode == 1:
        error_type = fault_list[int(start_trace_info[9])]
        error_list.append(error_type)
        category_list.append(category)
    
    if len(cur_childs) != 0:
        for child in cur_childs:
            child_line = ""
            for i in range(1, len(lines)):
                cur_line = lines[i]
                cur_info = cur_line.split(',')[span_pos]
                if cur_info == child:
                    child_line = cur_line
                    break
            if child_line != "":
                WorkForward(lines=lines, models=models, optimizers=optimizers, line=child_line, fault_list=fault_list, tmp=tmp, error_list=error_list, prev_out=cur_out, category_list=category_list, feature_list=feature_list, label_mode=label_mode)
    # It could represent that this span has come to end. Calculate the loss and backward
    else:
        if label_mode == 0:
            category_list.append(category)



def Logs2Vectors(cur_logs, tmp: TemplateMiner) -> list:
    """
    Transfer logs and spans to sentence vectors
    """
    sentences = cur_logs
    embedding = []
    outsentences = []
    if cur_logs != []:
        try:
            for i in range(len(sentences)):
                if sentences[i] == '':
                    continue
                sentence = RemoveSignals(tmp.add_log_message(sentences[i].strip())['template_mined'])
                outsentences.append(sentence)
            x, m = data_io.sentences2idx(sentences=outsentences, words=words)
            w = data_io.seq2weight(x, m, weight4ind)
            embedding = SIF_embedding.SIF_embedding(We, x, w, param)
        except Exception:
            pass
    return embedding

def GetValue(d: dict, v: str):
    v = v.split("-")[0]
    for key in d.keys():
        if v == d[key]:
            return key

 def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--servicelist", type=str, default="id_service.csv", help="csv file of service list.")
    parser.add_argument("--faultlist", type=str, default="id_fault2.csv", help="csv file of fault list.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size of training.")
    parser.add_argument("--trainset", type=str, default="", help="Abs path of training set")
    parser.add_argument("--labelmode", type=int, default=1, help="0=one label for one trace, 1=one label for one span")
    parser.add_argument("--drain", type=str, default="gaia", help="Name of drain model")
    parser.add_argument("--errortypes", type=int, default=72, help="Number of error types of the datset. The normal type should be the 0 one.")
    parser.add_argument("--fileconfig", type=str, default="2,4,9,10,11,12", help="Six numbers split by ',' to indicate the position of important information servicename, spanid, error code (if the dataset does not have errorcode, write any number is fine.), span event, child list and log list.")
    parser.add_argument("--train", type=bool, default=True, help="Indicate train or test mode.")
    parser.add_argument("--gloveword", type=str, default='./GloVeModel/vectors.txt', help="The word frequency file generated by GloVe")
    parser.add_argument("--glovevec", type=str, default='./GloVeModel/vocab.txt', help="The word vector file generated by GloVe.")
    
    # Set neccessary parameters
    global wordfile, weightfile, service_pos, span_pos, error_pos, event_pos, child_pos, log_pos,wordfile, weightfile, words, We, word2weight, weight4ind
    arguments = parser.parse_args()
    pre_config = arguments.fileconfig.split(',')
    service_pos = int(pre_config[0])
    span_pos = int(pre_config[1])
    error_pos = int(pre_config[2])
    event_pos = int(pre_config[3])
    child_pos = int(pre_config[4])
    log_pos = int(pre_config[5])
    
    wordfile = arguments.gloveword
    weightfile = arguments.glovevec
    (words, We) = data_io.getWordmap(wordfile)
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)
    
    # Start the main workflow
    fault_list = Init_workflow(arguments.faultlist)
    Sample(fault_list=fault_list, arguments=arguments)
        
main()
