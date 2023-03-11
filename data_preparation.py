import os
import sys
import time
import pandas as pd
import datetime
import logging

from tqdm import tqdm

# Import drain3 model
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.persistence_handler import PersistenceHandler

# Import Graph class
from TraceGraph import TraceGraph

data_dir = "../data/"

def Drain_Init():
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
    log_f = open(log_file, 'r')
    log_lines = log_f.readlines()
    log_f.close()

    logs = pd.DataFrame([], columns=['traceid', 'spanid', 'parentspan', 'childspans', 'service', 'loginfo'])
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
        span_record = {'traceid':[], 'spanid':[], 'parentspan':[], 'childspans':[], 'service':[], 'loginfo':[]}
        for i in tqdm(range(trace_lines.shape[0])):
            trace_line = trace_lines.loc[i]
            # Find a new trace
            if trace_line['TraceId'] != current_trace:
                current_trace = trace_line['TraceId']
                if len(span_record['traceid']) != 0:
                    for index in range(len(span_record['traceid'])):
                        if span_record['parentspan'][index] == '-1':
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
                    trace_logs.clear()
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
            # span_node = TraceGraph(trace_line['TraceId'], trace_line['SpanId'], trace_line['ParentSpan'])
            for log_dt in trace_logs:
                cur_span_id = trace_line['SpanId'][:-2]
                cur_start_time = trace_line['StartTime']
                cur_end_time = trace_line['EndTime']
                if log_dt[0] <= cur_end_time and log_dt[0] >= cur_start_time:
                    span_record['loginfo'][-1].append(log_dt)
                # Mine drain template
                tmp.add_log_message(log_dt[2])

            # Bound processing
            if i == trace_lines.shape[0] - 1 and trace_line['ParentSpan'] != "-1":
                for index in range(len(span_record['traceid'])):
                    if span_record['parentspan'][index] == '-1':
                        continue
                    span_record['childspans'][span_record['spanid'].index(span_record['parentspan'][index])].append(span_record['spanid'][index])
                # save to dataframe
                out = pd.DataFrame(span_record)
                logs = pd.concat([logs, out], axis=0)

        if os.path.exists(tgt_file):
            cache = pd.read_csv(tgt_file)
            logs = pd.concat([cache, logs], axis=0)
        logs.to_csv(tgt_file, index = False)
        tmp.save_state("tt")



def WorkFlow():
    """
    Data preparation work flow contorling.
    """
    # Create drain template miner.
    drain3_template = Drain_Init()
    data_dir = "./data/DeepTraLog/TraceLogData"
    tgt_dir = "./data/ProcessedData"

    tgt_file = ""
    # Read All data
    for f_0 in os.listdir(data_dir):
        if "F" in f_0:
            tgt_file = f_0[:3]
        else:
            tgt_file = "normal"
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
                TraceLogCombine(drain3_template, tf, lf, tgt_file)
            else:
                # If has no sub folder
                if f_3.endswith("csv"):
                    trace_files.append(f_3)
                else:
                    log_file = f_3
        if log_file != "":
            TraceLogCombine(drain3_template, trace_files, log_file, tgt_file)

# Tools
def Time2Timestamp(timestr: str):
    """
    Transfer time from format YYMMDD HHMMSS to timestamp
    """
    datetime_obj = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    timestamp = int(time.mktime(datetime_obj.timetuple()) * 1000 + datetime_obj.microsecond / 1000)
    return timestamp

def LogLineSplit(logline: str):
    infos = logline.split(' ')
    log_time = ' '.join(infos[0:2])
    log_timestamp = Time2Timestamp(log_time)


    cache = infos[2][9:-2].split(',')
    service_name = cache[0]
    trace_id = cache[2]
    span_id = cache[3]

    # serverity = infos[4]
    unstructure = ' '.join(infos[5:])
    return [log_timestamp, trace_id, unstructure]
    

if __name__ == '__main__':
    WorkFlow()
    test = pd.read_csv("test.csv")
    print(test)
    pass

