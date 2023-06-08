# Multi Agent Anomaly Detection
## Environment requirement
Linux. This system has been tested on Ubuntu 22.04.
# Python environment
python==3.7.1
torch==1.7.0+cu110
scikit-learn==1.0.2
# Other Requirement
Drain3, refer to https://github.com/logpai/Drain3.

GloVe, refer to https://github.com/stanfordnlp/GloVe.

SIF, refer to https://github.com/PrincetonML/SIF.
## Dataset
Please refer to https://github.com/FudanSELab/DeepTraLog to get the dataset.

or

Refer to http://docs.aiops.cloudwise.com/en/ to get another dataset.
## Quick Start
Use data_preparation.py to preprocess your dataset.

Use workflow_new.py to train multi agents.

Use SecondTimeClassification.py as the Multi-Decision Merger. 
