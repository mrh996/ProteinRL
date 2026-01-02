# -*- coding: utf-8 -*-
import torch
from time import time
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import random
import pickle
import os
from configs import get_cfg_defaults
from model_graph import Main_model
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
"""CPU or GPU."""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_dataset_txt(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            dataset.append({
                'wildtype_protein': line[0].lower(),
            })
    return dataset

def augment_graph(data, edge_drop_prob=0.2, node_mask_prob=0.2):
   
    edge_index = data.edge_index
    mask = torch.rand(edge_index.size(1)) > edge_drop_prob  
    edge_index_aug = edge_index[:, mask]

    x_aug = data.x.clone()
    mask_nodes = torch.rand(data.x.size(0)) < node_mask_prob
    x_aug[mask_nodes] += torch.randn_like(x_aug[mask_nodes])  

   
    edge_attr_aug = None
    if data.edge_attr is not None:
        edge_attr_aug = data.edge_attr[mask]

    data_aug = data.clone()
    data_aug.edge_index = edge_index_aug
    data_aug.x = x_aug
    if edge_attr_aug is not None:
        data_aug.edge_attr = edge_attr_aug

    return data_aug


def contrastive_loss(z1, z2, temperature=0.5):
   
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    similarity_matrix = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).long().to(z1.device)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

class ProteinDataset_withdata(Dataset):
    def __init__(self, data, protein_images):
        self.data = data
        self.protein_images = protein_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        wildtype_protein = row['wildtype_protein']

        
        if wildtype_protein not in self.protein_images:
            print(f"Protein ID not found: {wildtype_protein}")
            return None  


        wt_image = self.protein_images[wildtype_protein]


        wt_node_features = wt_image.x
        wt_edge_index = wt_image.edge_index
        wt_edge_attr = wt_image.edge_attr

        graph_wt = Data(x=wt_node_features, edge_index=wt_edge_index, edge_attr=wt_edge_attr)
        return graph_wt

class ProteinDataset(Dataset):
    def __init__(self, protein_images):
        
        self.protein_images = protein_images
        self.protein_ids = list(protein_images.keys())  # 提取所有蛋白质 ID

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        
        protein_id = self.protein_ids[idx]
        wt_image = self.protein_images[protein_id]

        
        wt_node_features = wt_image.x
        wt_edge_index = wt_image.edge_index
        wt_edge_attr = wt_image.edge_attr

        graph_wt = Data(x=wt_node_features, edge_index=wt_edge_index, edge_attr=wt_edge_attr)
        return graph_wt


def graph_to_data(graph):
    x = graph.x.clone().detach().to(torch.float)
    edge_index = graph.edge_index.clone().detach().to(torch.long)
    edge_attr = graph.edge_attr.clone().detach().to(torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            
            z1 = model(batch)

           
            batch_aug = augment_graph(batch)
            z2 = model(batch_aug)

           
            loss = contrastive_loss(z1, z2)
            total_loss += loss.item()

  
    return total_loss / len(val_loader)

def main():
    cfg = get_cfg_defaults()
    set_seed(cfg.MODEL.SEED)
    picklefile1 = cfg["DIR"]["picklefile1"]
    picklefile2 = cfg["DIR"]["picklefile2"]
    picklefile3 = cfg["DIR"]["picklefile3"]

    max_nodes = 0
    max_edges = 0
    protein_images = {}
    for file in [picklefile1]:
        protein_images.update(pickle.load(open(file, 'rb')))

    for name, graph in protein_images.items():
        if max_nodes < graph.x.shape[0]:
            max_nodes = graph.x.shape[0]
        if max_edges < graph.edge_index.shape[1]:
            max_edges = graph.edge_index.shape[1]

    data = read_dataset_txt(cfg["DIR"]["DATASET"])

    dataset = ProteinDataset_withdata(data, protein_images)

    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=False)

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Main_model(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')  
    best_model_path = cfg["DIR"]["OUTPUT_DIR"] + '/best_model.pth'

  
    import time

    for epoch in range(100):
        print(f"Epoch {epoch + 1}")
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            z1 = model(batch)
            batch_aug = augment_graph(batch)
            z2 = model(batch_aug)
            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

     
        val_loss = validate(model, val_loader, device)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")

    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with Val Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
