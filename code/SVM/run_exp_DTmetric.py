###################### Load DataSet File ###################################
import numpy as np
import pandas as pd
from helper import SpacyPreprocessor, countterm, process_commit_info
from helper import DependencyTree, FetureExtr, normalize_metrics
from helper import load_cross_validation_split 
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

use_filtered = False
n_fold = 10
labels, commits, metrics = load_cross_validation_split(use_filtered)

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
merged_commits, len_folds = merge(commits)
merged_metrics, _ = merge(metrics)

# Preprocess commit-info (text)
print("[1] normalize commit-info ...")
spacy_model = SpacyPreprocessor.load_model()
preprocessor = SpacyPreprocessor(spacy_model=spacy_model, remove_numbers=True, remove_stopwords=False, lemmatize=True, )
merged_commits = [preprocessor.preprocess_text(commit) for commit in merged_commits]
if not use_filtered:
    merged_commits = [process_commit_info(commit) for commit in merged_commits]
# Drive metrics (numerical) from commit-info
print("[2] add counting metrics from normalized commit-info ...")
for idx, commit in enumerate(merged_commits):
    char_count = len(commit)
    word_count = len(commit.split())
    word_density = char_count / (word_count+1)
    term_count = countterm(commit)
    merged_metrics[idx] += [char_count, word_count, word_density, term_count]
# Generate DT (dependency tree) and vectorize them
print("[3] translate normalized commit-info to DT, then to sngram vector ...")
merged_DTtexts = [DependencyTree(commit) for commit in merged_commits]
merged_X, merged_C = FetureExtr(merged_DTtexts, (1,1))
merged_X = pd.DataFrame.sparse.from_spmatrix(merged_X, columns=merged_C)
# Drive metrics (numerical) from DT
print("[4] add counting metrics from DT ...")
for idx, DTtext in enumerate(merged_DTtexts):
    sngram_count = len(DTtext.split())
    merged_metrics[idx].append(sngram_count)
# Normalize metrics and drive mean-metric (numerical)
print("[5] normalize all metrics via min-max and add mean-valud metric ...")
merged_metrics = normalize_metrics(merged_metrics)
mean_metric = np.mean(merged_metrics, axis=1)
merged_metrics = np.column_stack((merged_metrics, mean_metric))
# Concatenate DT vectors and metrics
print("[6] concate features (metrics) and sngram vectors ...")
merged_Xnew = np.concatenate((merged_X.values, merged_metrics), axis=1)
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
with open("../results/v5_10-folds_svm_both.json", "w") as f:
    json.dump(prob_dict, f)