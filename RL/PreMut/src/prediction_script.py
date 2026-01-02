import sys
import torch
sys.path.insert(0, "/LOCAL2/mur/MRH/protein_RL/PreMut/src")

from MyDataModule import MyDataModule
from LitModule import LitGat
import pytorch_lightning as pl
from constants import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import pandas as pd
from PDB_all_atom import PDBReader_All_Atom
from biopandas.pdb import PandasPdb
import pickle
from collections import defaultdict
import argparse
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, TransformerConv, aggr
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU

from dataset import *

from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing,GATConv, GATv2Conv, GraphUNet
from torch_geometric.nn import global_mean_pool, MeanAggregation
from egnn_clean import EGNN


class egnn_ablation_atom_types_only(torch.nn.Module):
    def __init__(self,in_channels: List,hidden_channels,num_classes,num_hidden_layers = NUM_HIDDEN_LAYERS) -> None:
        super().__init__()
        self.model = EGNN(in_node_nf=in_channels[1],hidden_nf=hidden_channels,out_node_nf=3,in_edge_nf=1,attention=True,n_layers=num_hidden_layers)

    def edge_index_to_edges(self,edge_index):
        edges = []
        edges.append(edge_index[0])
        edges.append(edge_index[1])
        return edges
    def forward(self,node_feats: tuple,edge_index, edge_attr,batch):
        # input = torch.cat((node_feats[1],node_feats[2]),dim=1)
        input = node_feats[1]
        edges = self.edge_index_to_edges(edge_index=edge_index)
        h,x = self.model(h=input,x=node_feats[0],edges=edges,edge_attr=edge_attr)
        return h

class predictionClass():
    def __init__(self,wild_pdb='1ert_A',mutation_info='D_59_N',input_pdb_dir='MutData2022_PDB',save_dir='predictions') -> None:
        self.wild_pdb = wild_pdb
        self.mutation_info = mutation_info
        self.input_pdb_dir = input_pdb_dir
        self.save_dir = save_dir
    def initialize_network(self):
        model = egnn_ablation_atom_types_only(in_channels=[3,37,21],hidden_channels=32,num_classes=3,num_hidden_layers=4)
        return model
    def initialize_Lit_module(self):
        net = self.initialize_network()
        lit_module = LitGat(MODEL=net,loss_weight=LOSS_WEIGHT)
        return lit_module
    def output_to_pdb(self,coords,node_one_hot_sequence_atoms,node_one_hot_sequence_residues,name='temp.pdb'):
        node_one_hot_sequence_atoms = node_one_hot_sequence_atoms.tolist()
        node_one_hot_sequence_residues = node_one_hot_sequence_residues.tolist()
        # print(len(node_one_hot_sequence_residues))
        residue_num = 1
        record_name = []
        atom_number = []
        blank_1 = []
        atom_name = []
        alt_loc = []
        residue_name = []
        blank_2 = []
        chain_id = []
        residue_number = []
        insertion = []
        blank_3 = []
        x_coord = []
        y_coord = []
        z_coord = []
        occupancy = []
        b_factor = []
        blank_4 = []
        segment_id = []
        element_symbol = []
        charge = []
        line_idx = []

        for idx, xyz in enumerate(coords):
            atom_type = ATOMS[node_one_hot_sequence_atoms[idx].index(1)]
            # residue_type = node_one_hot_sequence_residues[idx].index(1)
            try:
                residue_type = BASE_AMINO_ACIDS_3_LETTER[node_one_hot_sequence_residues[idx].index(1)]
            except:
                residue_type = 'X'
            if idx == 0:
                residue_num = 1

            else:
                try:
                    previous_residue_type = BASE_AMINO_ACIDS_3_LETTER[node_one_hot_sequence_residues[idx-1].index(1)]
                except:
                    previous_residue_type = 'X'

                # if residue_type != previous_residue_type:
                #     residue_num += 1
                if atom_type == 'N':
                    residue_num += 1

            # print('ATOM {0}  {4}  {5}  {6} {7}   {1} {2} {3} {8} {9}   {10}  {11}'.format(idx+1,xyz[0],xyz[1],xyz[2],atom_type,residue_type,'A',residue_num,'1.00','0.00',atom_type[0],idx+1))
            # with open('temp.pdb','a') as f:
            #     f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20}\n')
            record_name.append('ATOM')
            atom_number.append(idx+1)
            blank_1.append('')
            atom_name.append(atom_type)
            alt_loc.append('')
            residue_name.append(residue_type)
            blank_2.append('')
            chain_id.append('A')
            residue_number.append(residue_num)
            insertion.append('')
            blank_3.append('')
            x_coord.append(xyz[0])
            y_coord.append(xyz[1])
            z_coord.append(xyz[2])
            occupancy.append(1.00)
            b_factor.append(0.00)
            blank_4.append('')
            segment_id.append('')
            element_symbol.append(atom_type[0])
            charge.append(0.00)
            line_idx.append(idx+1)
        dict = {'record_name': record_name,'atom_number': atom_number,'blank_1': blank_1,'atom_name': atom_name, 'alt_loc': alt_loc, 'residue_name': residue_name,'blank_2': blank_2,'chain_id': chain_id
        ,'residue_number': residue_number, 'insertion': insertion,
        'blank_3': blank_3, 'x_coord': x_coord, 'y_coord': y_coord, 'z_coord': z_coord, 'occupancy': occupancy, 'b_factor': b_factor, 'blank_4': blank_4, 'segment_id': segment_id,
        'element_symbol': element_symbol, 'charge': charge, 'line_idx': line_idx}
    
        
        df = pd.DataFrame(dict)
        ppdb = PandasPdb()
        ppdb.df['ATOM'] = df
        # print(ppdb)
        ppdb.to_pdb(path=name,records=None,gz=False,append_newline=True)

    def predict(self):
        model = self.initialize_Lit_module()
        model_checkpoint_path = '/LOCAL2/mur/MRH/protein_RL/PreMut/Saved_Model/model.ckpt'
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"模型检查点文件不存在: {model_checkpoint_path}")

        if torch.cuda.is_available():
            checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cuda:0'))
        else:
            checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        
        net = model.model
        key_map = {}
        for key, value in checkpoint['state_dict'].items():
            new_key_name = 'model.' + key.split('model.')[-1]
            key_map[key] = new_key_name
        new_state_dict = {key_map.get(key, key): value for key, value in checkpoint['state_dict'].items()}

        net.load_state_dict(new_state_dict)
        pdbreader = PDBReader_All_Atom(pdb_dir=self.input_pdb_dir,mutant_pdb=self.wild_pdb,wild_pdb=self.wild_pdb,mutation_info=self.mutation_info,state='predict')
        
        graph, _ = pdbreader.pdb_to_graph()
        

        x= (graph.node_coords, graph.node_one_hot_sequence,graph.node_one_hot_sequence_residues)
        net.eval()
        out = net(node_feats=x, edge_index=graph.edge_index, edge_attr=graph.edge_distances,batch=graph.batch)
        
        predicted_name = self.wild_pdb.split('_')[0].lower()+'_A_'+self.mutation_info+'.pdb'
        out = out + graph.node_coords
        out_np = out.detach().clone().cpu().numpy()
        node_one_hot_sequence_np = graph.node_one_hot_sequence.detach().clone().cpu().numpy()
        node_one_hot_sequence_residues_np = graph.node_one_hot_sequence_residues.detach().clone().cpu().numpy()
        self.output_to_pdb(out_np,node_one_hot_sequence_atoms=node_one_hot_sequence_np,node_one_hot_sequence_residues=node_one_hot_sequence_residues_np,name=os.path.join(self.save_dir,predicted_name))


def main(wild_pdb_path,mutation_info,output_dir,chain_id):
    
    pdb_name=wild_pdb_path.split('.')[0].split('/')[-1]
    #print('pdb_name',pdb_name)
    if len(pdb_name.split('_'))>1:
        wild_pdb=pdb_name
    else:
        wild_pdb = pdb_name + '_' + chain_id
    

    tmp_lst = wild_pdb_path.split('.')[0].split('/')
    
    wild_pdb_dir = '/'.join(tmp_lst[:-1]) + '/'
    #print("wild_pdb is",wild_pdb)

    prediction = predictionClass(wild_pdb=wild_pdb, mutation_info=mutation_info, input_pdb_dir=wild_pdb_dir,
                                 save_dir=output_dir)
    prediction.predict()
    #print('wild_pdb is',wild_pdb)
    #print('mutation_info is',mutation_info)

    file_path = output_dir + '/' + wild_pdb.split('_')[0] + '_' +wild_pdb.split('_')[1]+'_'+ mutation_info + '.pdb'
    file_path_new=output_dir + '/' + wild_pdb.split('_')[0]+ '.pdb'
    if os.path.exists(file_path_new):
        #print(f"Deleting the previous mutated PDB file: {file_path_new}")
        os.remove(file_path_new)
    #print('file_path is', file_path)
    with open(file_path, 'r') as original_file, open(file_path + '.tmp', 'w') as temp_file:
        temp_file.write('HEADER    ' +  wild_pdb + '_' + mutation_info  + '\n')
        #temp_file.write('CRYST1   100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n')

        for line in original_file:
            temp_file.write(line)
        os.remove(file_path)
        os.rename(file_path + '.tmp', file_path_new)