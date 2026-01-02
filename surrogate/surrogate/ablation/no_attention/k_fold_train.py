# -*- coding: utf-8 -*-
import torch
from time import time
from torch import nn, optim
from torch.utils.data import Dataset, Subset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import KFold
import numpy as np
import random
import pickle
import copy
import os
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from configs import get_cfg_defaults
from model_ddg import Main_model
from tqdm import tqdm

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
                'wildtype_protein': line[0],
                'mutation_protein': line[0]+'_A_' + line[2] + '_' + line[5] + '_' + line[3],
                'label': float(line[4])
            })
    return dataset


class ProteinDataset(Dataset):
    def __init__(self, data, protein_images):
        self.data = data
        self.protein_images = protein_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        wildtype_protein = row['wildtype_protein']
        mutation_protein = row['mutation_protein']

        # Load image data
        wt_image = self.protein_images[wildtype_protein]
        mut_image = self.protein_images[mutation_protein]

        # Extract node features and edge features
        wt_node_features = wt_image.x
        wt_edge_index = wt_image.edge_index
        wt_edge_attr = wt_image.edge_attr

        mut_node_features = mut_image.x
        mut_edge_index = mut_image.edge_index
        mut_edge_attr = mut_image.edge_attr

        # Calculate difference image
        node_diff = wt_node_features - mut_node_features
        encoded_edges_mut = mut_edge_index[0] * max(wt_edge_index.max(), mut_edge_index.max()) + mut_edge_index[1]
        encoded_edges_wt = wt_edge_index[0] * max(wt_edge_index.max(), mut_edge_index.max()) + wt_edge_index[1]
        unique_edges = torch.cat([encoded_edges_wt, encoded_edges_mut]).unique(return_counts=True)
        unique_edges_mask = unique_edges[1] == 1
        unique_encoded_edges = unique_edges[0][unique_edges_mask]
        unique_edge_index = torch.stack([unique_encoded_edges // max(wt_edge_index.max(), mut_edge_index.max()), unique_encoded_edges % max(wt_edge_index.max(), mut_edge_index.max())])
        graph_diff = Data(x=node_diff, edge_index=unique_edge_index)

        graph_wt = Data(x=wt_node_features, edge_index=wt_edge_index, edge_attr=wt_edge_attr)
        label = row['label']
        label = torch.tensor(label, dtype=torch.float)
        return graph_wt, graph_diff, label

def read_graph_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        graph_data = pickle.load(f)
    return graph_data

def graph_to_data(graph):
    x = graph.x.clone().detach().to(torch.float)
    edge_index = graph.edge_index.clone().detach().to(torch.long)
    edge_attr = graph.edge_attr.clone().detach().to(torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def main():
    cfg = get_cfg_defaults()
    set_seed(cfg.MODEL.SEED)

    image_folder = cfg["DIR"]["GRAPH"]
#    all_files = os.listdir(image_folder)
#    image_files = [file for file in all_files if file.endswith('.pkl')]

    max_nodes = 0
    max_edges = 0


    protein_images = pickle.load(open(os.path.join(image_folder,'large_dictionary.pickle'), 'rb'))
    for name, graph in protein_images.items():
        if max_nodes < graph.x.shape[0]:
            max_nodes = graph.x.shape[0]
        if max_edges < graph.edge_index.shape[1]:
            max_edges = graph.edge_index.shape[1]
#    for image_file in image_files:
#        protein_name = image_file.split('.')[0]
#        graph = pickle.load(open(os.path.join(image_folder, image_file), 'rb'))
#        protein_images[protein_name] = graph_to_data(graph)
#        if max_nodes < protein_images[protein_name].x.shape[0]:
#            max_nodes = protein_images[protein_name].x.shape[0]
#        if max_edges < protein_images[protein_name].edge_index.shape[1]:
#            max_edges = protein_images[protein_name].edge_index.shape[1]

#    dataFolder = cfg.DIR.DATAFOLDER
    data = read_dataset_txt(cfg["DIR"]["DATASET"])
    dataset = ProteinDataset(data, protein_images)

    # Splitting the dataset into training+validation and test sets
    num_test = int(len(dataset) * 0.1)  # 10% of data for testing
    num_train_val = len(dataset) - num_test
    train_val_dataset, test_dataset = random_split(dataset, [num_train_val, num_test])

    test_loader = DataLoader(test_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.MODEL.SEED)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'This is fold number: {fold+1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=False)


        model = Main_model(**cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.MODEL.LR)
        trainer = Trainer(model, opt, device, train_loader, val_loader, test_loader, **cfg)
        fold_result = trainer.train()
        results.append(fold_result)
        print(f'fold number: {fold + 1}, results are:  {fold_result}')
    average_result = np.mean(results, axis=0)
    with open(os.path.join(cfg.DIR.OUTPUT_DIR, "model_architecture.txt"), "w+") as wf:
        wf.write(str(model))
        wf.write(f"Average results across folds: {average_result}")
    print(f"Directory for saving result: {cfg.DIR.OUTPUT_DIR}")

    return average_result

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["MODEL"]["NUM_EPOCHS"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = config["MODEL"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_r2_score = -float('inf')
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_rmse_epoch = []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["DIR"]["OUTPUT_DIR"]

        train_metric_header = ["# Epoch", "Train_loss"]
        valid_metric_header = ["# Epoch", "rmse", "r2", "Val_loss"]

        self.train_table = PrettyTable(train_metric_header)
        self.val_table = PrettyTable(valid_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            rmse, r2_score_val, val_loss = self.test(dataloader="val")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [rmse, r2_score_val, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_rmse_epoch.append(rmse)

            if r2_score_val >= self.best_r2_score: # or rmse <= self.best_rmse_score:
                self.best_model = copy.deepcopy(self.model)
                self.best_r2_score = r2_score_val
                # self.best_rmse_score = rmse
                self.best_epoch = self.current_epoch
                self.save_model(self.model, self.output_dir)

            with open(os.path.join(self.output_dir, 'realtime_monitoring.txt'), 'a') as file:
                file.write('Validation at Epoch {}: Loss: {}, R2 Score: {}, RMSE Score: {}\n'.format(self.current_epoch, val_loss, r2_score_val, rmse))
            print('Validation at Epoch {}: Loss: {}, R2 Score: {}, RMSE Score: {}'.format(self.current_epoch, val_loss, r2_score_val, rmse))

        self.test_metrics = self.test()
        self.save_final()
        return self.test_metrics

    def save_final(self):
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'a') as fp:
            fp.write(self.val_table.get_string())
        with open(train_prettytable_file, "a") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        criterion = nn.MSELoss()
        for i, (wt_graph, diff_graph, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            wt_graph, diff_graph = wt_graph.to(device), diff_graph.to(device)
            labels = torch.tensor(labels, dtype=torch.float, device=self.device)
            self.optim.zero_grad()
            score = self.model(wt_graph, diff_graph)
            loss = criterion(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch {}: Loss: {}'.format(self.current_epoch, loss_epoch))

        return loss_epoch


    def test(self, dataloader = "test"):
        self.model.eval()
        test_loss = 0.0
        all_labels = []
        all_predictions = []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        with torch.no_grad():
            criterion = nn.MSELoss()
            for i, (wt_images, diff_images, labels) in enumerate(data_loader):
                wt_images, diff_images = wt_images.to(device), diff_images.to(device)
                labels = torch.tensor(labels, dtype=torch.float, device=self.device)

                outputs = self.model(wt_images, diff_images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

            test_loss = test_loss / len(self.test_dataloader)
            r2_score_test = r2_score(all_labels, all_predictions)
            mae = mean_absolute_error(all_labels, all_predictions)
            rmse = np.sqrt(mae ** 2)

        if dataloader == "test":
            test_metrics = {
                "test_loss": test_loss,
                "r2_score": r2_score_test,
                "mae": mae,
                "rmse": rmse,
                "best_epoch": self.best_epoch
            }
            return test_metrics

        if dataloader == "val":
            return rmse, r2_score_test, test_loss

    def save_model(self, model, dir):
        filename = dir + '/model'
        torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

