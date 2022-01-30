from tqdm import tqdm

import torch

from treelstm import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        # tree_set, sent_set, label_set, metric_set = dataset[0], dataset[1], dataset[2], dataset[3]

        metric_set = dataset[1]
        dataset = dataset[0]
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, sentence, label = dataset[indices[idx]]
            metric_input = metric_set[idx]
            target = utils.map_label_to_target(label, dataset.num_classes)
            sentence = sentence.to(self.device)
            target = target.to(self.device)
            metric_input = metric_input.to(self.device)
            output = self.model(tree, sentence, metric_input)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        metric_set = dataset[1]
        dataset = dataset[0]
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(0, dataset.num_classes, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                tree, input, label = dataset[idx]
                metric_input = metric_set[idx]
                target = utils.map_label_to_target(label, dataset.num_classes)
                input = input.to(self.device)
                target = target.to(self.device)
                metric_input = metric_input.to(self.device)
                output = self.model(tree, input, metric_input)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss / len(dataset), predictions
