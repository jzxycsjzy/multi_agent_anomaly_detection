# Multi Agent Anomaly Detection
## Environment requirement
Linux. This system has been tested on Ubuntu 22.04, python3.7.1.
# Python environment
Use the requirements.txt as follow.
```
conda create -n MAAD python=3.7
conda activate MAAD
pip install -r requirements.txt
```
# Other Requirement
Drain3, refer to https://github.com/logpai/Drain3.

GloVe, refer to https://github.com/stanfordnlp/GloVe.

SIF, refer to https://github.com/PrincetonML/SIF.
## Dataset
Please refer to https://github.com/FudanSELab/DeepTraLog to get the dataset.

or

Refer to http://docs.aiops.cloudwise.com/en/ to get another dataset.
## Quick Start
Firstly, split the whole dataset into saperated trace file such as example_log.txt and put all these file in a certain file such as "./data/train/". Furthermore, if the dataset has trace-level labels, the name of each trace file shoulb be like traceid_faulttype.* (e.g. 7bf800940ab64c55a70add01ad6b847b.37.16284749994970479_71.csv). And ensure to generate fault list file such as id_fault.csv and service3 name list such as id_service.csv.

Use MAADWorkflow.py to train multi agents.
```python
python MAADWorkflow.py --servicelist id_service.csv --faultlist id_fault.csv --batch 1 --trainset ./data/train/ --labelmode 0 --errortypes 72 --train True
```

When the parameter ```train``` is set as False, the program will become inference mod. And the program could generate an MAADout.txt which contains a series of multi-agent confidence lists and its corresponding labels.

Use Multi=decision_Merger.py as the Multi-Decision Merger. 
```python
python Multi_decision_Merger.py --trainset MAADout_test.txt
```
