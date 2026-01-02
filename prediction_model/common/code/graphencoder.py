import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import contactmap_features as cmf
from model_ddg import Main_model
from model_ddg import ProteinGraphConvModule
import numpy as np
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as utils
import math
from torch_geometric.nn import global_mean_pool
import os

from configs_pre import get_cfg_defaults
from configs import get_cfg_defaults as get_model_cfg

def graph_to_data(graph):
    """
    save the graph to data
    """
    x = graph.x.clone().detach().to(torch.float)
    edge_index = graph.edge_index.clone().detach().to(torch.long)
    edge_attr = graph.edge_attr.clone().detach().to(torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)  # Assuming all nodes are in the same graph
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    return data


def graph_encoder(Graph_wt, cfg):
    wt_graph_data = graph_to_data(Graph_wt)
    cfg_model = get_model_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model_path = os.path.join(cfg.DDG_MODEL.DIR, 'model')
    model = Main_model(**cfg_model).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Extract the state_dict of protein_gnn only
    protein_gnn_state_dict = model.protein_gnn.state_dict()

    # Initialize pretrained encoder and load only the protein_gnn weights
    pretrained_encoder = ProteinGraphConvModule(cfg_model["PROTEIN"]["NUM_FEATURE_NODE"],
                                                cfg_model["GNN"]["DIM"],
                                                cfg_model["GNN"]["DEPTH"],
                                                cfg_model["GNN"]["HEAD"]).to(device)
    pretrained_encoder.load_state_dict(protein_gnn_state_dict, strict=False)
    pretrained_encoder.eval()

    # Convert graph data to device
    wt_graph_data = wt_graph_data.to(device)

    with torch.no_grad():
        # Get updated node features
        updated_node_features = pretrained_encoder(wt_graph_data.x, wt_graph_data.edge_index, wt_graph_data.batch)

    # 创建与 Graph_wt 一致的新图结构并更新节点特征
    updated_graph = Graph_wt.clone()  # 假设 Graph_wt 有一个 clone() 方法可以复制其结构
    updated_graph.x = updated_node_features  # 替换为更新后的节点特征

    return updated_graph


# Example usage:
if __name__ == '__main__':
    cfg = get_cfg_defaults()

    wild_name = '3f6r'
    wild_pdb_path = cfg.PROTEIN.WT_FOLDER + '/' + wild_name + '.pdb'

    therhold = cfg.PROTEIN.THERSHOLD

    Graph_wt, seq_wt = cmf.generate_protein_features(wild_pdb_path, wild_name, therhold)

    updated = graph_encoder(Graph_wt,cfg)

    print(updated)
    # predict_ddg is the reward