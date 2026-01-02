from torch.nn.modules.module import Module
from functions.custom_func import JaggedLogSoftmax, JaggedArgmax, JaggedMax
import networkx as nx
import numpy as np
from torch import nn

class JaggedLogSoftmaxModule(nn.Module):
    def __init__(self):
        super(JaggedLogSoftmaxModule, self).__init__()
        self.jagged_log_softmax = JaggedLogSoftmax.apply

    def forward(self, logits, prefix_sum):
        return self.jagged_log_softmax(logits, prefix_sum)

class JaggedArgmaxModule(nn.Module):
    def __init__(self):
        super(JaggedArgmaxModule, self).__init__()
        self.jagged_argmax = JaggedArgmax.apply

    def forward(self, values, prefix_sum):
        return self.jagged_argmax(values, prefix_sum)

class JaggedMaxModule(nn.Module):
    def __init__(self):
        super(JaggedMaxModule, self).__init__()
        self.jagged_max = JaggedMax.apply

    def forward(self, values, prefix_sum):
        return self.jagged_max(values, prefix_sum)
