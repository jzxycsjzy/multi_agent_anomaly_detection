# Multi Agent Anomaly Detection
## Environment requirement
Linux. This system has been tested on Ubuntu 22.04.
# Python environment
```
python==3.7.1
torch==1.7.0+cu110
scikit-learn==1.0.2
```
or just use the requirements.txt
# Other Requirement
Drain3, refer to https://github.com/logpai/Drain3.

GloVe, refer to https://github.com/stanfordnlp/GloVe.

SIF, refer to https://github.com/PrincetonML/SIF.
## Dataset
Please refer to https://github.com/FudanSELab/DeepTraLog to get the dataset.

or

Refer to http://docs.aiops.cloudwise.com/en/ to get another dataset.
## Quick Start
Firstly, split the whole dataset into saperated trace file such as example_log.txt and put all these file in a certain file such as "./data/train/". And ensure to generate fault list file such as id_fault.csv and service3 name list such as id_service.csv.

Use workflow_new.py to train multi agents.
```python
python MAADWorkflow.py --servicelist id_service.csv --faultlist id_fault.csv --batch 1 --trainset ./data/train/ --labelmode 0 --errortypes 72
```

Use SecondTimeClassification.py as the Multi-Decision Merger. 
```python
python Multi_decision_Merger.py --trainset train.txt --testset test.txt
```
