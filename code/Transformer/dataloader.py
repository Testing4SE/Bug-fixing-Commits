import pandas as pd
import numpy as np
import random
import json

def load_data(filepath, sheetname=None):
    if sheetname is None:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_excel(filepath, sheet_name=sheetname)

    # get label
    labels = list(df['flag'])
    assert not any(pd.isna(labels))

    # get nl input
    nl_inputs = list(df['commit-info'])
    assert not any(pd.isna(nl_inputs))

    # get feat input
    other_inputs = []
    for column_name in df.columns.values:
        if column_name not in ['number', 'author', 'data', 'flag', 'commit-info']:
            column_value = list(df[column_name])
            other_inputs.append(column_value)
            assert not any(pd.isna(column_value))
    other_inputs = np.asarray(other_inputs).T.tolist()
    
    assert len(labels)==len(nl_inputs)==len(other_inputs)
    return labels, nl_inputs, other_inputs

def load_all_data():
    excel_fn = '../data/datasets-withMectrics-v5.xlsx'
    sheet_names = ['math', 'guava', 'guice', 'gumtree', 'jedis', 'commons-io', 'commons-imaging', 'commons-pool',
                   'httpcomponents-client', 'commons-email', 'mockito', 'closure-compiler', 'commons-text', 'joda-time', 'gson']
    
    Y = []
    X_NL = []
    X_FEAT = []
    for sheet_name in sheet_names:
        y, x_nl, x_feat = load_data(excel_fn, sheet_name)
        for yi, xi_nl, xi_feat in zip(y, x_nl, x_feat):
            Y.append(yi)
            X_NL.append(xi_nl)
            X_FEAT.append(xi_feat)

    assert len(Y)==len(X_NL)==len(X_FEAT)
    print('[Data Size]', len(Y))
    return Y, X_NL, X_FEAT

def shuffle_data(labels, sentences, features):
    n = len(labels)
    ids = random.sample(range(n), n)
    labels = [labels[idx] for idx in ids]
    sentences = [sentences[idx] for idx in ids]
    features = [features[idx] for idx in ids]
    return labels, sentences, features

def cross_validation_split(labels, sentences, features, k=10):
    n = len(labels)
    label_folds, sentence_folds, feature_folds = [], [], []
    fold_size = (n-1) // k + 1
    while labels!=[]:
        label_folds.append(labels[:fold_size])
        labels = labels[fold_size:]
        sentence_folds.append(sentences[:fold_size])
        sentences = sentences[fold_size:]
        feature_folds.append(features[:fold_size])
        features = features[fold_size:]
    print('[Fold Size]', [len(fold) for fold in label_folds])
    return label_folds, sentence_folds, feature_folds

def load_cross_validation_split(use_filtered):
    if use_filtered==True:
        #fname = "../data/v4_10-folds_filtered.json" # This is the old version split
        assert False, "NOT support!"
    else:
        #fname = "../data/v4_10-folds.json"
        fname = "../data/v5_10-folds.json"
    with open(fname, "r") as f:
        d = json.load(f)
        labels_folds = d["label"]
        commits_folds = d["commit"]
        metrics_folds = d["metric"]
        return labels_folds, commits_folds, metrics_folds

if __name__=="__main__":

    # Create new 10-fold split
    random.seed(1234)
    labels, commits, metrics = load_all_data()
    labels, commits, metrics = shuffle_data(labels, commits, metrics)
    labels_folds, commits_folds, metrics_folds = cross_validation_split(labels, commits, metrics, 10)

    # DO NOT un-comment this. 
    # Already saved two versions and the code for old version is modified and lost.
    #dump_dict = {"label": labels_folds, "commit": commits_folds, "metric": metrics_folds}
    #with open("../data/v4_10-folds.json", "w") as f:
    #    json.dump(dump_dict, f)

    # Compare with old 10-fold split
    labels_folds_, commits_folds_, metrics_folds_ = load_cross_validation_split()
    for labels, commits, metrics, labels_, commits_, metrics_ in zip(labels_folds, commits_folds, metrics_folds, labels_folds_, commits_folds_, metrics_folds_):
        assert labels==labels_
        assert commits==commits_
        assert metrics==metrics_
