import os
import json

precision_data_list = []
recall_data_list = []
path = 'commit_log/'

total_dump_dict = {}

fold = 10
for fold in range(fold):
    dir = path
    precision_list = []
    recall_list = []
    fold_dump_dict = {}
    for j in range(10):
        # print("epoch: ", str(j))
        file = dir + 'data' + str(fold) + '/total' + str(j) + '.txt'
        predicts = []
        labels = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                labels.append(int(float(line[0])))
                predict = float(line[1])
                predicts.append(predict)
        fold_dump_dict['epoch_' + str(j + 1)] = predicts

    total_dump_dict['fold_' + str(fold + 1)] = fold_dump_dict

with open("v5_10-folds_treelstm_both.json", "w") as f:
    json.dump(total_dump_dict, f)
