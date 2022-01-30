###################### Load DataSet File ###################################
import numpy as np
import pandas as pd
from helper import normalize_metrics
from helper import load_cross_validation_split 
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

use_filtered = False
n_fold = 10
labels, _, metrics = load_cross_validation_split(use_filtered)

##################### Data Processing #####################################
def merge(datas):
    lens = [len(data) for data in datas]
    merged_data = []
    for data in datas:
        merged_data += data
    return merged_data, lens

def unmerge(merged_data, lens):
    datas = []
    for l in lens:
        datas.append(merged_data[:l])
        merged_data = merged_data[l:]
    assert len(merged_data)==0
    return datas

# MERGE k-folds
merged_metrics, len_folds = merge(metrics)

# Normalize metrics and drive mean-metric (numerical)
print("[5] normalize all metrics via min-max and add mean-valud metric ...")
merged_metrics = normalize_metrics(merged_metrics)
mean_metric = np.mean(merged_metrics, axis=1)
merged_metrics = np.column_stack((merged_metrics, mean_metric))
merged_Xnew = merged_metrics
print("=> shape", merged_Xnew.shape)

# UNMERGE k-folds
X = unmerge(merged_Xnew, len_folds)
Y = [np.asarray(fold_labels) for fold_labels in labels]

##################### K-fold Cross Validation #####################################
prob_dict = {}
for fold_idx in range(n_fold):
    # Prepare split
    X_train = np.concatenate((X[:fold_idx]+X[fold_idx+1:]), axis=0)
    Y_train = np.concatenate((Y[:fold_idx]+Y[fold_idx+1:]), axis=0)
    X_test = X[fold_idx]
    Y_test = Y[fold_idx]
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')
    # Train the model
    clf.fit(X_train, Y_train)
    # Predict on test set
    #Y_pred = clf.predict(X_test)
    Y_prob = clf.decision_function(X_test)
    Y_pred = [1 if prob>0.5 else 0 for prob in Y_prob]
    prob_dict["fold_{}".format(fold_idx+1)] = Y_prob.tolist()
    # Statistics
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    print("[fold {}] acc {:.4f} precision {:.4f} recall {:.4f}".format(fold_idx+1, acc, prec, recall))
with open("../results/v5_10-folds_svm_metric.json", "w") as f:
    json.dump(prob_dict, f)