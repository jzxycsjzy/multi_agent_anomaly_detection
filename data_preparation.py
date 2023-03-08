import os
import sys
import logging

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
    drain3Config.load('./drain3/drain3/ini')
    drain3Config.profiling_enabled = True

    tmp = TemplateMiner(config=drain3Config, persistence_handler=ph)
    return tmp

def TraceLogCombine(tmp: TemplateMiner):
    """
    params:
        tmp: drain3 template miner object

    TODO:
        1. Read log data and trace data and combine them together.
            Final data structure should be like:
                trace_id.csv = span_id: str, child_span: list[str], service_id: int, unstructure_data: str
                Tips: it will be better if there is a function could be used for checking DAG.
        2. drain3 template will be mined at the same time.
    """         
    data_dir = "data/TraceLogData"
    pass

def Unzip_Data():
    os.system("cd data/TraceLogData")
    for i in range(1, 15):
        s = "%02d" % i
        os.system("pwd")
        os.system("cd data/TraceLogData; ./unzip.sh " + s)

def WorkFlow():
    """
    Data preparation work flow contorling.
    """
    # Create drain template miner.
    drain3_template = Drain_Init()


if __name__ == '__main__':
    Unzip_Data()

