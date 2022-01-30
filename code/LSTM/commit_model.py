import torch
import torch.nn as nn
import torch.nn.functional as F

from treelstm import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for distance-angle similarity
class Merge(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Merge, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.l1 = nn.Linear(self.mem_dim, self.hidden_dim)
        self.l2 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.num_classes)
        self.l4 = nn.Linear(24, self.hidden_dim)

    def forward(self, tree_state, metric_inputs):
        tree_state = self.l1(tree_state)
        metric_state = self.l4(metric_inputs)
        vec = torch.cat((tree_state, metric_state), 1)
        
        out = F.sigmoid(self.l2(vec))
        out = F.log_softmax(self.l3(out), dim=1)
        return out



class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.merge = Merge(mem_dim, hidden_dim, num_classes)

    def forward(self, tree, inputs, metric_inputs):
        inputs = self.emb(inputs)
        state, hidden = self.childsumtreelstm(tree, inputs)
        output = self.merge(state, metric_inputs)
        return output

