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


class DiffGraphGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_layers, output_dim=128):
        super(DiffGraphGNN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)

    def forward(self, features_a, features_b):
        query = self.query_projection(features_a)
        key = self.key_projection(features_b)
        value = self.value_projection(features_b)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (features_a.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, value)

        return attended_features + features_a, attention_weights


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Main_model(nn.Module):
    def __init__(self, **config):
        super(Main_model, self).__init__()
        node_feature_dim = config["PROTEIN"]["NUM_FEATURE_NODE"]
        hidden_dim = config["GNN"]["DIM"]
        GNN_depth = config["GNN"]["DEPTH"]
        k_head = config["GNN"]["HEAD"]
        out_channels = config["GNN"]["OUT_CHANNEL"]
        hidden_dim2 = config["GNN"]["DIM2"]
        num_layers = config["MODEL"]["MLP_LAYER"]

        self.protein_gnn = ProteinGraphConvModule(node_feature_dim, hidden_dim, GNN_depth, k_head)
        self.cross_attention = CrossAttention(hidden_dim)
        self.diff_gnn = DiffGraphGNN(node_feature_dim, hidden_dim, GNN_depth)
        self.f_final = nn.Linear(hidden_dim, hidden_dim2)

        self.f_mlp = MLP(hidden_dim2*2, hidden_dim2, hidden_dim2, num_layers)

        self.prediction = nn.Linear(hidden_dim2, out_channels)

    def forward(self, data_wt, data_diff):
        
        batch_index_wt = data_wt.batch
        batch_index_diff = data_diff.batch

        x_wt, edge_index_wt = data_wt.x, data_wt.edge_index
        x_diff, edge_index_diff = data_diff.x, data_diff.edge_index

       
        vertex_feature_wt = self.protein_gnn(x_wt, edge_index_wt, batch_index_wt)
        vertex_feature_diff = self.diff_gnn(x_diff, edge_index_diff)

        
        graph_feature_wt = global_mean_pool(vertex_feature_wt, batch_index_wt)
        graph_feature_diff = global_mean_pool(vertex_feature_diff, batch_index_diff)

        fused_features, att_weights = self.cross_attention(graph_feature_wt, graph_feature_diff)

        
        combined_features = torch.cat([fused_features, graph_feature_diff], dim=-1)

        out_mlp = self.f_mlp(combined_features)

        out = self.prediction(out_mlp)
        out = out.squeeze(-1)

        return out, att_weights
