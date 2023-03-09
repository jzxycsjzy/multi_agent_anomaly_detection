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

    for trace_file in trace_list:
        trace_lines = pd.read_csv(trace_file)
        trace_id = ""
        count = 0
        logs = pd.DataFrame([], columns=['traceid', 'index', 'spanid', 'parentspanid', 'loginfo'])
        logs.loc[logs.shape[0]] = [0,1,2,3,4]
        print(logs)
        for i in tqdm(range(trace_lines.shape[0])):
            line = trace_lines.loc[i]
            
        exit()



def WorkFlow():
    """
    Data preparation work flow contorling.
    """
    # Create drain template miner.
    drain3_template = Drain_Init()
    data_dir = "./data/DeepTraLog/TraceLogData"
    tgt_dir = "./data/PreprocessedData"

    tgt_file = ""
    # Read All data
    for f_0 in os.listdir(data_dir):
        if "F" in f_0:
            tgt_file = f_0[:3]
        else:
            tgt_file = "normal"
        tgt_file = os.path.join(tgt_dir, tgt_file)
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
    pass

if __name__ == '__main__':
    WorkFlow()
    pass

