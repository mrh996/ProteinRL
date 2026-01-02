import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as utils
import math
from torch_geometric.nn import global_mean_pool

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_length=6000):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings

        
        pe = torch.zeros(max_length, num_embeddings)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_embeddings, 2).float() * -(math.log(10000.0) / num_embeddings))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        pe_adjusted = self.pe[:, :x.size(1), :]
        return x + pe_adjusted

class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads):
        super(CustomConv, self).__init__(aggr='add')
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.lin = nn.Linear(in_channels, out_channels)
        self.query = nn.Linear(out_channels, out_channels)
        self.key = nn.Linear(out_channels, out_channels)
        self.value = nn.Linear(out_channels, out_channels)

        self.attention_bias = nn.Parameter(torch.Tensor(1, num_heads))
        nn.init.xavier_normal_(self.attention_bias)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index_i, x_i, x_j):
        
        query = self.query(x_i).view(-1, self.num_heads, self.head_dim)
        key = self.key(x_j).view(-1, self.num_heads, self.head_dim)
        value = self.value(x_j).view(-1, self.num_heads, self.head_dim)

        
        attention_scores = (query @ key.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_scores = attention_scores + self.attention_bias.unsqueeze(-1)
        attention_scores = F.softmax(attention_scores, dim=-1)

        
        out = attention_scores @ value
        out = out.view(-1, self.num_heads * self.head_dim)
        return out

class ProteinGraphConvModule(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, GNN_depth, k_head, output_dim=64):
        super(ProteinGraphConvModule, self).__init__()
        assert hidden_dim % k_head == 0, "hidden_dim must be divisible by k_head"
        assert output_dim % k_head == 0, "output_dim must be divisible by k_head"

        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(GNN_depth - 1): 
            self.gnn_layers.append(CustomConv(hidden_dim, hidden_dim, k_head))
        self.gnn_layers.append(CustomConv(hidden_dim, output_dim, k_head))  

        self.pos_encoder = PositionalEncodings(num_embeddings=output_dim)

        self.super_feature_update = nn.GRUCell(output_dim, output_dim)
        self.super_feature_transform = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch_index):
        x = F.relu(self.node_embedding(x))

        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))  
        super_feature = torch.zeros(batch_index.max().item() + 1, x.size(1), device=x.device)
        for i in range(batch_index.max().item() + 1):
            mask = (batch_index == i)
            super_feature[i] = x[mask].mean(dim=0)

        super_feature = F.relu(self.super_feature_transform(super_feature))  
        h0 = torch.zeros(super_feature.size(), device=super_feature.device)  
        for i in range(super_feature.size(0)):
            updated_feature = self.super_feature_update(super_feature[i], h0[i])
            super_feature = super_feature.clone()
            super_feature[i] = updated_feature

        x = x.unsqueeze(0)
        x = self.pos_encoder(x)
        x = x.squeeze(0)

        expanded_super_feature = super_feature[batch_index]

  
        graph_representation = torch.cat([x, expanded_super_feature], dim=1)
        return graph_representation


class Main_model(nn.Module):
    def __init__(self, **config):
        super(Main_model, self).__init__()
        node_feature_dim = config["PROTEIN"]["NUM_FEATURE_NODE"]
        hidden_dim = config["GNN"]["DIM"]
        GNN_depth = config["GNN"]["DEPTH"]
        k_head = config["GNN"]["HEAD"]

        self.protein_gnn = ProteinGraphConvModule(node_feature_dim, hidden_dim, GNN_depth, k_head)

        self.pool = global_mean_pool

    def forward(self, data):
        batch_index = data.batch
        x, edge_index = data.x, data.edge_index


        vertex_feature = self.protein_gnn(x, edge_index, batch_index)

        graph_embedding = self.pool(vertex_feature, batch_index)
        return graph_embedding