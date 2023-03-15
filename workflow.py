from __future__ import annotations
from typing import Tuple

import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

from memory_profiler import profile

from MAADModel import MAADModel

def Init_model() -> list[MAADModel]:
    """
    Init model from service list. Assign one model for each service
    """
    model_list = {}
    service_list = pd.read_csv("id_service.csv").drop(labels="Unnamed: 0", axis=1)
    # service_list = service_list.sample(frac=1).reset_index(drop=True)
    # print(service_list.loc[0])
    for i in range(service_list.shape[0]):
        model_list[service_list.loc[i]["Service"]] = MAADModel()
    return model_list

def Init_workflow() -> Tuple[list[pd.DataFrame], dict]:
    fault_list = pd.read_csv("id_fault.csv", index_col="id")
    src_path = 'data/ProcessedData/'
    pds = []
    for i in tqdm(range(len(fault_list.shape[0]))):
        f = os.path.join(src_path, fault_list.loc[i]['filename'])
        dt = pd.read_csv(f)
        pds.append(dt)
    return pds, fault_list.to_dict()['filename']

def Sample(models: list[MAADModel], pds: list[pd.DataFrame], fault_list):
    steps = [0] * 15
    epoch = 0
    while epoch < 15:
        for i in range(15):
            cur_pd = pds[i]
            count = 0
            start_index = steps[i]
            end_index = steps[i]
            while steps[i] < cur_pd.shape[0]:
                if cur_pd[steps[i]]['ParentSpan'] == '-1' or cur_pd[steps[i]]['ParentSpan'] == -1:
                    count += 1
                    if count == 2:
                        break
                    end_index = steps[i]
                    steps[i] += 1
            

        

if __name__ == '__main__':
    models = Init_model()
    pds = Init_workflow()
    Sample(models=models, pds=pds)