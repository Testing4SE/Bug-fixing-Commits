import random
import json
import os


def load_cross_validation_split():
    fname = "data/v5_10-folds.json"
    with open(fname, "r") as f:
        d = json.load(f)
        labels_folds = d["label"]
        commits_folds = d["commit"]
        metrics_folds = d["metric"]
        return labels_folds, commits_folds, metrics_folds


if __name__ == "__main__":
    random.seed(1234)
    labels_folds, commits_folds, metrics_folds = load_cross_validation_split()

    for fold_idx in range(10):
        label_eval, commit_eval, metric_eval = labels_folds[fold_idx], commits_folds[fold_idx], metrics_folds[fold_idx]
        label_train, commit_train, metric_train = [], [], []
        for idx in range(10):
            if idx != fold_idx:
                label_train += labels_folds[idx]
                commit_train += commits_folds[idx]
                metric_train += metrics_folds[idx]
        path = "commit_data/data" + str(fold_idx)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/train.txt', 'w') as f1, open(path + '/train_metric.txt', 'w') as f2:
            f1.write("1\t2\t3\t4\t5\n")

            for i in range(len(label_train)):
                f1.write(
                    str(i) + "\t" + str(commit_train[i].replace('\t', ' ')) + "\t" + "hello" + "\t" + str(
                        label_train[i]) + "\t" + "hello\n")
                f2.write(' '.join([str(metric) for metric in metric_train[i]]) + "\n")
        with open(path + '/test.txt', 'w') as f1, open(path + '/test_metric.txt', 'w') as f2:
            f1.write("1\t2\t3\t4\t5\n")

            for i in range(len(label_eval)):
                f1.write(
                    str(i) + "\t" + str(commit_eval[i].replace('\t', ' ')) + "\t" + "hello" + "\t" + str(
                        label_eval[i]) + "\t" + "hello\n")
                f2.write(' '.join([str(metric) for metric in metric_eval[i]]) + "\n")
