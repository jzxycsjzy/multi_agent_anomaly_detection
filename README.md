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

or any other dataset and ensure that the dataset has at least trace data.
## Quick Start
Firstly, prepate the data at first. It is NECESSARY to do the things below at the same time.
1. Split the whole dataset into saperated trace files, each filename format is traceid_faulttype.* (e.g. 7bf800940ab64c55a70add01ad6b847b.37.16284749994970479_71.csv) and put them into the same folder.
2. Generate faults list as id_fault.csv and services list as id_service.csv.
3. Generate drain3 model and use this model to construct corpus to train a GloVe model.

Use MAADWorkflow.py to train multi agents.
```python
python MAADWorkflow.py --servicelist id_service.csv --faultlist id_fault.csv --batch 1 --trainset ./data/train/ --labelmode 0 --errortypes 72 --train True --gloveword ./GloVeModel/vectors.txt --glovevec ./GloVeModel/vocab.txt --drain trainticket --fileconfig "4,1,0,6,3,5"
```

When the parameter ```train``` is set as False, the program will become inference mod. And the program could generate an MAADout.txt which contains a series of multi-agent confidence lists and its corresponding labels.

Then, use Multi_decision_Merger.py as the Multi-Decision Merger. 
```python
python Multi_decision_Merger.py --trainset MAADout_test.txt
```
