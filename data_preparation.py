from __future__ import annotations

import os
import sys
import time
# import modin
import pandas as pd
# import modin.pandas as pd
import datetime
import logging
import random

from multiprocessing import Process
from tqdm import tqdm

# Import drain3 model
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.persistence_handler import PersistenceHandler

# import modin.config as modin_cfg
# import ray
# # Import Graph class
# from TraceGraph import TraceGraph

# # Import NLP model
# from SIF import params, data_io, SIF_embedding

# # Init SIF parameters
# wordfile = "data/glove/vectors.txt" # word vector file, can be downloaded from GloVe website
# weightfile = "data/glove/vocab.txt" # each line is a word and its frequency
# weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
# rmpc = 1 # number of principal components to remove in SIF weighting scheme
# # load word vectors
# (words, We) = data_io.getWordmap(wordfile)
# # load word weights
# word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
# weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# # set parameters
# param = params.params()
# param.rmpc = rmpc



def Drain_Init() -> TemplateMiner:
    """
    Create drain3 model
    """
    ph = PersistenceHandler()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    drain3Config = TemplateMinerConfig()
    drain3Config.load('./config/drain3.ini')
    drain3Config.profiling_enabled = True

    tmp = TemplateMiner(config=drain3Config, persistence_handler=ph)
    return tmp

def TraceLogCombine(tmp: TemplateMiner, trace_list: list, log_file: str, tgt_file: str):
    """
    params:
        tmp: drain3 template miner object
        trace_list: list of trace files
        log_list: list of log files
        tgt_file: file to save trace-log data

    TODO:
        1. Read log data and trace data and combine them together.
            Final data structure should be like:
                trace_id.csv = span_id: str, child_span: list[str], service_id: int, unstructure_data: str
                Tips: it will be better if there is a function could be used for checking DAG.
        2. drain3 template will be mined at the same time.
    """         
    print("######################")
    print(trace_list, log_file, tgt_file)
    with open("finished_log2.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == log_file:
            return
    log_f = open(log_file, 'r')
    log_lines = log_f.readlines()
    log_f.close()

    # Load Drain model
    if os.path.exists("DrainModel/tt"):
        tmp.load_state("tt")

    logs = pd.DataFrame([], columns=['traceid', 'spanid', 'parentspan', 'childspans', 'service', 'loginfo', 'spaninfo'])
    current_trace = ""
    trace_logs = []
    

    for trace_file in trace_list:
        trace_lines = pd.read_csv(trace_file)
        # logs.loc[logs.shape[0]] = ["a001", "a001.1", '-', ['a001.2', 'a001.3'], ['abc', 'def']]
        # print(type(logs))
        # print(type(logs.loc[0]))
        # a = logs.loc[0]
        # print(a['log_info'])
        # exit()
        # Store all span data
        tgt_file = tgt_file.replace('test', 'train')
        span_record = {'traceid':[], 'spanid':[], 'parentspan':[], 'childspans':[], 'service':[], 'loginfo':[], 'spaninfo':[]}
        for i in tqdm(range(trace_lines.shape[0])):
            trace_line = trace_lines.loc[i]
            # Find a new trace
            if trace_line['TraceId'] != current_trace:
                current_trace = trace_line['TraceId']
                if len(span_record['traceid']) != 0:
                    for index in range(len(span_record['traceid'])):
                        if span_record['parentspan'][index] == '-1' or span_record['parentspan'][index] == -1:
                            continue
                        span_record['childspans'][span_record['spanid'].index(span_record['parentspan'][index])].append(span_record['spanid'][index])
                    # save to dataframe
                    out = pd.DataFrame(span_record)
                    logs = pd.concat([logs, out], axis=0)
                    # clear all previous data
                    span_record['traceid'].clear()
                    span_record['spanid'].clear()
                    span_record['parentspan'].clear()
                    span_record['childspans'].clear()
                    span_record['loginfo'].clear()
                    span_record['service'].clear()
                    span_record['spaninfo'].clear()
                    trace_logs.clear()
                    if i > trace_lines.shape[0] * 0.7 and 'test' not in tgt_file:
                        if os.path.exists(tgt_file):
                            cache = pd.read_csv(tgt_file)
                            logs = pd.concat([cache, logs], axis=0)
                        logs.to_csv(tgt_file, index=False)
                        tgt_file = tgt_file.replace('train', 'test')
                        logs = pd.DataFrame([], columns=['traceid', 'spanid', 'parentspan', 'childspans', 'service', 'loginfo', 'spaninfo'])
                remove_logs = []

                for j in range(len(log_lines)):
                    log_line = log_lines[j]
                    # timestamp, traceid, spanid, infos
                    structured_log = LogLineSplit(log_line)
                    if current_trace == structured_log[1]:
                        trace_logs.append(structured_log)
                        remove_logs.append(log_line)
                # Logs will be remove from the log list after they have been processed.
                for line in remove_logs:
                    log_lines.remove(line)
                
                
            # Existing trace, new span
            # Process Trace and logs, then create trace event graph(TEG)
            # print(trace_logs)
            
            span_record['traceid'].append(trace_line['TraceId'])
            span_record['spanid'].append(trace_line['SpanId'])
            span_record['parentspan'].append(trace_line['ParentSpan'])
            span_record['childspans'].append([])
            span_record['service'].append(trace_line['Service'])
            span_record['loginfo'].append([])
            span_record['spaninfo'].append(trace_line['URL'])
            # span_node = TraceGraph(trace_line['TraceId'], trace_line['SpanId'], trace_line['ParentSpan'])
            for log_dt in trace_logs:
                cur_span_id = trace_line['SpanId'][:-2]
                cur_start_time = trace_line['StartTime']
                cur_end_time = trace_line['EndTime']
                if log_dt[3] == cur_span_id:
                    if log_dt[0] <= cur_end_time and log_dt[0] >= cur_start_time:
                        span_record['loginfo'][-1].append(log_dt)
                # Mine drain template
                tmp.add_log_message(log_dt[2].strip())
                tmp.add_log_message(trace_line['URL'])

            # Bound processing
            if i == trace_lines.shape[0] - 1 and trace_line['ParentSpan'] != "-1":
                for index in range(len(span_record['traceid'])):
                    if span_record['parentspan'][index] == '-1' or span_record['parentspan'][index] == -1:
                        continue
                    span_record['childspans'][span_record['spanid'].index(span_record['parentspan'][index])].append(span_record['spanid'][index])
                # save to dataframe
                out = pd.DataFrame(span_record)
                logs = pd.concat([logs, out], axis=0)


        if os.path.exists(tgt_file):
            cache = pd.read_csv(tgt_file)
            logs = pd.concat([cache, logs], axis=0)
        print(tgt_file)
        logs.to_csv(tgt_file, index = False)
        # tmp.save_state("tt2")

    # with open("finished_log2.txt", 'a+') as f:
    #     f.write(log_file + "\n")



def WorkFlow():
    """
    Data preparation work flow contorling.
    """
    # Multi-processing init
    process_list = []
    parameter_list = []
    # Create drain template miner.
    drain3_template = Drain_Init()
    data_dir = "./data/DeepTraLog/TraceLogData"
    tgt_dir = "./data/ProcessedData2"

    tgt_file = ""
    # Read All data
    for f_0 in os.listdir(data_dir):
        if "F" in f_0:
            tgt_file = "train_" + f_0
        else:
            tgt_file = "train_normal"
        tgt_file = os.path.join(tgt_dir, tgt_file + '.csv')
        log_file = ""
        trace_files = []
        f_1 = os.path.join(data_dir, f_0)
        for f_2 in os.listdir(f_1):
            f_3 = os.path.join(f_1, f_2)
            # If has sub folder
            if os.path.isdir(f_3):
                tf = []
                lf = ""
                for f_4 in os.listdir(f_3):
                    f_5 = os.path.join(f_3, f_4)
                    if f_4.endswith("csv"):
                        tf.append(f_5)
                    else:
                        lf = f_5
                parameter_list.append((drain3_template, tf, lf, tgt_file))
                # p = Process(target=TraceLogCombine, args=(drain3_template, tf, lf, tgt_file))
                # process_list.append(p)
                # TraceLogCombine(drain3_template, tf, lf, tgt_file)
            else:
                # If has no sub folder
                if f_3.endswith("csv"):
                    trace_files.append(f_3)
                else:
                    log_file = f_3
        if log_file != "":
            # TraceLogCombine(drain3_template, trace_files, log_file, tgt_file)
            parameter_list.append((drain3_template, trace_files, log_file, tgt_file))
            # p = Process(target=TraceLogCombine, args=(drain3_template, tf, lf, tgt_file))
            # process_list.append(p)
    for para in parameter_list:
        p = Process(target=TraceLogCombine, args=para)
        process_list.append(p)
    running_process = 0
    inrunning = []
    while True:
        if running_process == len(process_list):
            break
        if len(inrunning) < 50:
            print("added new process: {}".format(running_process))
            process_list[running_process].start()
            # process_list[running_process].join()
            inrunning.append(running_process)
            running_process += 1
        finish = []
        for index in inrunning:
            if not process_list[index].is_alive():
                finish.append(index)
        for p in finish:
            inrunning.remove(p)
        finish.clear()

# Tools
def Time2Timestamp(timestr: str):
    """
    Transfer time from format YYMMDD HHMMSS to timestamp
    """
    datetime_obj = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    timestamp = int(time.mktime(datetime_obj.timetuple()) * 1000 + datetime_obj.microsecond / 1000)
    return timestamp

def LogLineSplit(logline: str):
    """
    Format the log line data and return some useful data
    """
    infos = logline.split(' ')
    log_time = ' '.join(infos[0:2])
    log_timestamp = Time2Timestamp(log_time)
    cache = infos[2][9:-2].split(',')
    service_name = cache[0]
    trace_id = cache[2]
    span_id = cache[3]

    # serverity = infos[4]
    unstructure = ' '.join(infos[5:])
    return [log_timestamp, trace_id, unstructure, span_id]

def GloveCorpusConstruction():
    """
    Generate world vector from templates
    """
    save_dir = "data/glove/corpus"
    tmp = Drain_Init()
    tmp.load_state("tt2")
    save_file = open(save_dir, 'a+')
    for cluster in tmp.drain.clusters:
        template = cluster.get_template()
        save_file.write(RemoveSignals(template))
    save_file.close()

def RemoveSignals(line: str):
    """
    Remove all signals, numbers and single alpha from log line
    """
    remove_list = list("~`!@#$%^&*()-_=+[{]};:'\",<.>/?|\\0123456789")
    res = line
    for signal in remove_list:
        res = res.replace(signal, ' ')
    res_list = res.split()
    alpha = "abcdefghijklmnopqrstuvwxyz"
    alpha_upper = alpha.upper()
    alpha_lower = alpha.lower()
    alpha_list = list(alpha_upper + alpha_lower)
    for a in alpha_list:
        while a in res_list:
            res_list.remove(a)
    res = ' '.join(res_list)
    return res

def Ini_Workflow():
    fault_list = pd.read_csv("id_fault2.csv", index_col="id")
    src_path = '/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/ProcessedData2/test'
    pds = []
    for i in tqdm(range(fault_list.shape[0])):
        f = os.path.join(src_path, "test_" + fault_list.loc[i]['filename'] + '.csv')
        if not os.path.exists(f):
            print("file not exists!")
            continue
        dt = pd.read_csv(f)
        pds.append(dt)
    return pds, fault_list.to_dict()['filename']


def workflow2(pds: list[pd.DataFrame], fault_list: dict):
    print(len(pds))
    process_list = []
    for i in range(72):
        p = Process(target=single_process, args=(pds[i], i))
        p.start()
        process_list.append(p)

def single_process(pd: pd.DataFrame, fault_type: int):
    save_dir = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/ProcessedData/test/"
    count = 0

    start_index = 0
    end_index = 0
    
    trace_count = 0
    for i in tqdm(range(pd.shape[0])):
        if pd.loc[i]['parentspan'] == '-1' or pd.loc[i]['parentspan'] == -1 or i == pd.shape[0] - 1:
            count += 1
            if count == 2:
                trace_count += 1
                end_index = i
                if end_index == pd.shape[0] - 1:
                    end_index += 1
                pd.iloc[start_index: end_index].to_csv(save_dir + pd.loc[start_index]['traceid'] + "_" + str(fault_type) + ".csv", index=False)
                start_index = end_index
                count = 1


if __name__ == '__main__':
    # TraceLogCombine(Drain_Init(), ['/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/F01-01/SUCCESSF0101_SpanData2021-08-14_10-22-48.csv'], '/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/F01-01/F0101raw_log2021-08-14_10-22-51.log', "train_F01.csv")
    # WorkFlow()
    # GloveCorpusConstruction()

    # pds, fault_list = Ini_Workflow()
    # workflow2(pds, fault_list)
    tmp = Drain_Init()
    trace = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/F01-01/SUCCESSF0101_SpanData2021-08-14_10-22-48.csv"
    log = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/DeepTraLog/TraceLogData/F01-01/F0101raw_log2021-08-14_10-22-51.log"


    filesize = os.stat(trace).st_size + os.stat(log).st_size 
    all_span = pd.read_csv(trace)
    trace_list = []
    for i in tqdm(range(all_span.shape[0])):
        line = all_span.loc[i]
        if line['ParentSpan'] == "-1" or line['ParentSpan'] == -1:
            trace_list.append(line['TraceId'])
    
    second_size = 0
    tgt_path = "/home/rongyuan/workspace/anomalydetection/multi_agent_anomaly_detection/data/ProcessedData"
    type_dt = ["train", "test"]
    for t in type_dt:
        file_path = os.path.join(tgt_path, t)
        for file in os.listdir(file_path):
            if file.split('_')[0] in trace_list:
                f = os.path.join(file_path, file)
                second_size += os.stat(f).st_size
    print(second_size, filesize)
