from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier # 0.84
from sklearn.metrics import recall_score, precision_score

from tqdm import tqdm
import argparse

def CheckData(file = "res.txt"):
    data = []
    labels = []
    src_file = file
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
        data.append(eval(info[3]))
        labels.append(int(info[1]))
        categories = eval(info[3])
        # if i == 1000:
        #     break
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
        res = res.index(max(res))
        if res == label:
            res_dict[label][0] += 1
    r = [0,0,0]
    for c in res_dict.keys():
        res_dict[c].append(res_dict[c][0] / res_dict[c][1])
        for i in range(2):
            r[i] += res_dict[c][i]
    r[2] = r[0] / r[1]
    maxlen = 0
    for item in data:
        if len(item) > maxlen:
            maxlen = len(item)
    real_dt = []
    for item in data:
        real_dt.append([])
        for confidences in item:
            real_dt[-1] += confidences
        for i in range(maxlen - len(item)):
            real_dt[-1] += [0] * 72
    for i in range(len(labels)):
        labels[i] = 0 if labels[i] != 71 else 1 # 0-abnormal 1-normal
    return real_dt, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--trainset", type=str, default="train.txt", help="trainset of Multi-decision Merger")
    parser.add_argument("--testset", type=str, default="test.txt", help="testset of Multi-decision Merger")
    arguments = parser.parse_args()
    
    data, labels = CheckData(arguments.trainset)
    train_data, train_label = CheckData(arguments.testset)
    # Split trainset and test set
    X_train, _, y_train, _ = train_test_split(data, labels, test_size=0.05, random_state=1002)
    X_test, _, y_test, _ = train_test_split(train_data, train_label, test_size=0.05, random_state=1002)
    # Train the model
    clf2 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, l2_regularization=0.2, min_samples_leaf=4, verbose=1, max_leaf_nodes=200, early_stopping=False)
    clf2.fit(X_train, y_train)
    scores = clf2.score(X_test, y_test)
    scores.mean()
    res = clf2.predict(X_test)
    r = recall_score(y_test, res, pos_label=0)
    p = precision_score(y_test, res, pos_label=0)
    print('Accuracy:', scores)
    print('Recall: ', r)
    print('Precision:', p)

