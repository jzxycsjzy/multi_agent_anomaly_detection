import os
import sys

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss

import numpy as np
from tqdm import tqdm
import pandas as pd

from MAADModel2 import CategoryModel

__stdout__ = sys.stdout # 标准输出就用这行
sys.stdout = open('test_data.txt', 'a+')

class ResDataset(Dataset):
    def __init__(self, data_file, transforms = None) -> None:
        super().__init__()
        with open(data_file, 'r') as f:
            self.data = f.readlines()
        # self.data = [line.split(";")[1:] for line in self.data]
        # self.labels = [int(x[0]) for x in self.data]
        # self.categories = [eval(x[1]) for x in self.data]
        self.transforms = transforms

        self.fault_list_0 = pd.read_csv("id_fault.csv", index_col="id").to_dict()['filename']
        self.fault_list_1 = pd.read_csv("id_fault2.csv", index_col="id").to_dict()['filename']

        # self.maxseqlength = max([len(x) for x in self.categories])

    def __getitem__(self, index):
        sample = {}
        line = self.data[index]
        info = line.split(';')
        label1 = int(info[1])
        if self.fault_list_1[label1] == 'normal':
            second_label = 0
        else:
            second_label = int(self.fault_list_1[label1].split("-")[1]) - 1
        sample['label_0'] = second_label
        sample['label_1'] = int(info[2])
        sample['categories'] = eval(info[3])
        # data = self.categories[index]
        # label = self.labels[index]
        # seq_len =len(data)
        # pad_data = np.zeros(shape=(self.maxseqlength, 72))
        # pad_data[:seq_len] = data
        # sample ={'data': pad_data, 'label': label, 'seq_len': seq_len}
        return sample
    
    def __len__(self):
        return len(self.data)

def Data_Collection():
    src_file = "log.txt"
    tgt_file = "res.txt"
    with open(src_file, 'r') as f:
        lines = f.readlines()
    save_file = open(tgt_file, "w+")
    for line in lines:
        if line[0] == "#":
            save_file.write(line)
    save_file.close()

    with open(tgt_file, 'r') as f:
        lines = f.readlines()
    count = 0
    correct = 0
    for i in tqdm(range(len(lines))):
        line = lines[i]
        count += 1
        info = line.split(";")
        label = int(info[1])
        if "total" in info[3]:
            info[3] = info[3][:info[3].index("total")]
        if "Time" in info[3]:
            info[3] = info[3][:info[3].index("Time")]
        categories = eval(info[3])
        res = [0] * 72
        max_confidence = -1
        max_vote = -1
        for category in categories:
            c = category.index(max(category))
            if max(category) > max_confidence:
                max_confidence = max(category)
                max_vote = c
            # res.append(c)
            res[c] += 1
        res[max_vote] += 5
        if label == res.index(max(res)):
            correct += 1
        print(label, res)
    print("{}/{} = {}".format(correct, count, correct/count))

def collate_fn(batch):
    category_list = [item['categories'] for item in batch]
    labels = [item['label_0'] for item in batch]
    labels = torch.tensor(labels, dtype=torch.long)
    labels_1 = [item['label_1'] for item in batch]
    return {
        'categories': category_list,
        'label': labels,
        'label_1': labels_1
    }

# if __name__ == '__main__':
#     # Data_Collection()
#     dataset = ResDataset("res.txt")
#     dataloader = DataLoader(dataset, 1, True, collate_fn=collate_fn)
#     print("Finish dataloader.")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     models = []
#     optimizers = []
#     for i in range(15):
#         model = CategoryModel()
#         model.to(device)
#         models.append(model)
#         optimizer = Adam(model.parameters(), lr=0.001)
#         optimizers.append(optimizer)
#     loss_func = CrossEntropyLoss()
#     count = 0
#     correct = 0

#     isFault = torch.tensor([1], dtype=torch.long)
#     isFault = isFault.to(device)

#     for epoch in range(45):
#         print("Epoch")
#         for sample in dataloader:
#             dt = torch.tensor(sample['categories'],dtype=torch.float32)
#             dt = dt.to(device)
#             label = sample['label']
#             r = label.numpy().tolist()[0]
#             label = label.to(device)

#             label_1 = sample["label_1"]
#             label_model = label_1[0]
#             # dt = data['categories']
#             # dt = torch.tensor(dt)
#             # tgt = data['label']
#             # seq_len = [d.shape[0] for d in dt]
#             # seq_len = torch.tensor(seq_len)
#             # for d in dt:
#             #     print(d.shape[0])
#             # dt = pack_padded_sequence(dt, seq_len, batch_first=True, enforce_sorted=False)
#             out = models[label_model](dt)
#             loss = loss_func(out, label)
#             if label_model != 14:
#                 normal_out = models[14](dt)
#                 loss_normal = loss_func(normal_out, isFault)
#                 loss_normal.backward()
#                 optimizers[14].step()
#                 optimizers[14].zero_grad()
#             # loss = loss / dt.shape[1]
#             # if dt.shape[1] == 1:
#             #     loss = loss * 18
#             loss.backward()
#             res = out.cpu().detach().numpy().tolist()[0]
#             r_ = res.index(max(res))
#             print(label_model, r, r_, res)
#             count += 1
#             if r == r_:
#                 correct += 1
#             if count == 300:
#                 print(correct/count)
#                 correct = 0
#                 count = 0
#             optimizers[label_model].step()
#             optimizers[label_model].zero_grad()
#             # unpacked = pad_packed_sequence(out)
#             # out, bz = unpacked[0], unpacked[1]
#             # print(out, bz)
#         for i in range(15): 
#             torch.save(models[i].state_dict(), "./cateogryModel/category_" + str(i) + ".pt")

def CheckData():
    src_file = "res.txt"
    with open(src_file, 'r') as f:
        lines = f.readlines()
    res_dict = {}
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if line[0] != "#":
            continue
        info = line.split(";")
        label = int(info[1])
        if label not in res_dict.keys():
            res_dict[label] = [0, 1] # correct, total
        else:
            res_dict[label][1] += 1
        if "total" in info[3]:
            info[3] = info[3][:info[3].index("total")]
        if "Time" in info[3]:
            info[3] = info[3][:info[3].index("Time")]
        categories = eval(info[3])
        res = [0] * 72
        for i in range(len(categories)):
            # r = category.index(max(category))
            # res.append(r)
            category = categories[i]
            cache = sorted(category)
            for item in cache:
                index = category.index(item)
                c = 1
                if i == len(categories) - 1:
                    c = 5
                res[index] += (item % 0.1) * c

            # add_num = 1
            
            #     add_num = 12
            # res[category] += add_num
        res = res.index(max(res))
        if res == label:
            res_dict[label][0] += 1
    r = [0,0,0]
    for c in res_dict.keys():
        res_dict[c].append(res_dict[c][0] / res_dict[c][1])
        for i in range(2):
            r[i] += res_dict[c][i]
    r[2] = r[0] / r[1]
    print(r)
    print(res_dict)
CheckData()
# Data_Collection()
        