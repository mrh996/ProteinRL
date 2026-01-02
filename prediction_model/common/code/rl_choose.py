import argparse
import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from contactmap_features import load_pdb
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import pickle as cp
from configs_pre import get_cfg_defaults
from prediction_ddg import mutation_caller,predict_reward
from contactmap_features import generate_protein_features
cmd_opt = argparse.ArgumentParser(description='Argparser locally')
cmd_opt.add_argument('-mlp_hidden', type=int, default=64, help='mlp hidden layer size')
cmd_opt.add_argument('-att_embed_dim', type=int, default=64, help='att_embed_dim')
cmd_opt.add_argument('-num_steps', type=int, default=10000, help='# fits')
local_args, _ = cmd_opt.parse_known_args()
from configs import get_cfg_defaults as get_model_cfg
from graphencoder import graph_encoder
print(local_args)

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import S2VGraph
from cmd_args import cmd_args
from dnn import GraphClassifier

sys.path.append('%s/../../data_generator' % os.path.dirname(os.path.realpath(__file__)))
from data_util import load_pkl

sys.path.append('%s/../../graph_classification' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset
cfg = get_cfg_defaults()
def load_npy(file_path):
    return np.load(file_path, allow_pickle=True)

def parse_mutation_info(mutation_info):
    parts = mutation_info.split('_')
    
    return parts[0], int(parts[1]), parts[2]

def get_value_from_mutation(array, name, mutation_info):
    target_name = name
    target_wt, target_pos, target_mut = parse_mutation_info(mutation_info)
    
    for entry in array:
        entry_name, pos, wt, mut, value = entry[0], int(entry[5]), entry[2], entry[3], float(entry[4])
        if entry_name == target_name and pos == target_pos and wt == target_wt and mut == target_mut:
            return value
    return None

class GraphEdgeEnv(object):
    def __init__(self,predictor):
        self.classifier = predictor

    def setup(self, g_list):
        self.wild_names=[]
        self.data=load_npy('/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/large_data.npy')
        self.first_nodes = None
        self.n_steps = 0
        self.mutation_info=[]
        self.selected_nodes = None
        
        self.banned_list = None
        self.nodes = []
        self.new_struct=[]
        self.structure_list=[]
        self.possible_table = ['A', 'R', 'N', 'D', 'C', 'Q', 'E',  'G', 'H', 'L', 'I', 'K',  'M', 'F','P',  'S', 'T',  'V',  'W', 'Y']
        self.cfg = get_cfg_defaults()
        self.n_nodes = 0
        self.graph_list=[]
        self.protein_sequences=[]
        self.wild_pdb_path=[]
        self.prefix_sum=[]
        self.rm = cfg.PROTEIN.TEMP_RM
        self.threshold = cfg.PROTEIN.THERSHOLD
        self.temp=cfg.PROTEIN.TEMP
        self.multi_reward_threshold=cfg.THERSHOLD
        self.maximum_changes= cfg.STEPS
        
        for i in range(len(g_list)):
            graph, name, sequence, path=g_list[i]
            self.graph_list.append(graph)
            self.wild_pdb_path.append(path)
            self.protein_sequences.append(sequence)
            self.nodes.append(graph.num_nodes)
            self.wild_names.append(name)
            self.n_nodes += graph.num_nodes
            self.prefix_sum.append(self.n_nodes)
        self.pdb_path=self.wild_pdb_path
        self.rewards = np.zeros_like(np.arange(len(self.graph_list)))
        
        

    def bannedActions(self, g, node_x):        
        comps = [c for c in nx.connected_component_subgraphs(g)]
        set_id = {}
        for i in range(len(comps)):
            for j in comps[i].nodes():
                set_id[j] = i

        banned_actions = set()
        for i in range(len(g)):
            if set_id[i] != set_id[node_x] or i == node_x:
                banned_actions.add(i)
        return banned_actions

    def step(self, actions):  
         
        if self.first_nodes is None:  # pick the first node of edge
            #print('####start first action#####')
            
            assert self.n_steps % 2 == 0
            self.first_nodes = actions
            self.banned_list = []
            for i in range(len(self.protein_sequences)):
                selected_node_name = self.protein_sequences[i][self.first_nodes[i]]
                self.mutation_info.append(str(selected_node_name)+'_'+str(self.first_nodes[i]))
                selected_protein = self.protein_sequences[i][self.first_nodes[i]]
                self.banned_list.append(set([self.possible_table.index(str(selected_protein)),
                                        *(range(20,self.graph_list[i].num_nodes))]))
            
            print(f"Updated banned_list length: {len(self.banned_list)}")
            
                    
        else:  # mutation picked 
            old_pdb_paths = []  # Store old paths to remove later
            new_graph_list = []  # Store new graphs
            new_pdb_paths = []  # Store new paths
            #print('####start second action#####')
            for i in range(len(actions)):
                
                # Store old path if it's not the wild type
                if self.pdb_path[i] != self.wild_pdb_path[i]:
                    old_pdb_paths.append(self.pdb_path[i])
                    
                # Generate mutation
                
                new_type = self.possible_table[actions[i]]
                node=int(self.mutation_info[i].split('_')[1])
                
                self.protein_sequences[i] = self.protein_sequences[i][:node] + new_type + self.protein_sequences[i][node+1:]
                self.mutation_info[i] = self.mutation_info[i]+'_'+new_type
                
                # Calculate reward
                #print('Processing mutation for pdb:', self.pdb_path[i],'i th', i)
                new_re, new_graph, new_path =predict_reward(
                    self.pdb_path[i], 
                    self.wild_names[i],
                    self.mutation_info[i], 
                    self.cfg
                )
                self.rewards[i] = self.rewards[i] + new_re
                
                # Generate new mutation
                
                
                new_graph_list.append(new_graph)
                new_pdb_paths.append(new_path)
                
            #print('#### end second action#####')    
            # Update all paths and graphs after successful mutations
            self.graph_list = new_graph_list
            self.pdb_path = new_pdb_paths
            
            #print('####reward#####', self.rewards)
            '''
            # Clean up old PDB files after we're done using them
            for old_path in old_pdb_paths:
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        print(f"Removed old PDB file: {old_path}")
                    except Exception as e:
                        print(f"Error removing file {old_path}: {e}")
            '''
                        
            self.first_nodes = None
            self.banned_list = None
            self.mutation_info=[]
            
            
        self.n_steps += 1

        
            

    def uniformRandActions(self):
        act_list = []
        offset = 0
        #self.banned_list=[]
        #print(self.nodes)
        print('len(self.prefix_sum)',len(self.prefix_sum))
        for i in range(len(self.prefix_sum)):
            n_nodes = self.prefix_sum[i] - offset
            
            if self.first_nodes is None:
                act=np.random.randint(n_nodes)
                act_list.append(act)
                #selected_protein=self.protein_sequences[i][act]
                #self.banned_list.append(set([self.possible_table.index(str(selected_protein)),*(range(20, n_nodes))]))
            else:   
                print('self.banned_list length',len(self.banned_list)) 
                banned_actions = self.banned_list[i]
                #print('banned_actions',banned_actions)
                valid_indices = list(set(range(n_nodes)) - banned_actions)
                
                #valid_indices = [i for i in range(len(self.possible_table)) if self.possible_table != banned_actions]
                act_2=random.choice(valid_indices)
               
                act_list.append(act_2)
                
            offset = self.prefix_sum[i]
        return act_list

    def sampleActions(self, probs, greedy=False):
        offset = 0
        act_list = [] 
        p_list=[]
        p_values=[]
        for i in range(len(self.prefix_sum)):
            p_vec = probs[offset : self.prefix_sum[i], 0].to(dtype=torch.float64)

            if self.first_nodes is not None:
                banned_actions = self.banned_list[i]
                for j in banned_actions:
                    p_vec[j] = 0.0
                assert len(banned_actions) < len(p_vec)

            p_vec = p_vec / sum(p_vec)
            p_vec = p_vec.detach().numpy()
            p_values.append(p_vec)
            if greedy:
                action = np.argmax(p_vec)
            else:
                action = np.argmax(np.random.multinomial(1, p_vec))
            act_list.append(action)
            offset = self.prefix_sum[i]

        return act_list,p_values

    def isTerminal(self):
        if self.rewards.all() > self.multi_reward_threshold or self.n_steps == 2*self.maximum_changes:
            return True
        return False

    def getStateRef(self):
        
        cp_first = [None] * len(self.wild_pdb_path)
        if self.first_nodes is not None:
            cp_first = self.first_nodes
        b_list = [None] * len(self.wild_pdb_path)
        if self.banned_list:
            b_list = self.banned_list   
        
        # Ensure all lists have the same length
        if len(self.graph_list) == len(cp_first) == len(b_list):
            cur_list = zip(self.graph_list, cp_first, b_list)
            cur_list = list(cur_list)  # Convert to a list to inspect the contents
            #print(f'Contents of cur_list: {cur_list}')
        else:
            print('The lengths of the lists do not match!')
        return cur_list
    
    def cloneState(self):
        
        cp_first = [None] * len(self.graph_list)
        if self.first_nodes is not None:
            cp_first = self.first_nodes[:]
        b_list = [None] * len(self.graph_list)
        if self.banned_list is not None:
            b_list = self.banned_list[:]
        
        return list(zip(deepcopy(self.graph_list), cp_first, b_list))
    
def load_model(model_path, model_config, device):
    """
    Load the ddg prediction model from a file.
    """
    model = Main_model(**model_config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model 

def load_graphs(path_list,frac_train = 0.9):
    cfg = get_cfg_defaults()
    cfg_model = get_model_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    num_train = int(frac_train * len(path_list))

    train_glist = []
    test_glist = []
    label_map = {}
    for i in range(len(path_list)):
        graph_list, sequence_list = generate_protein_features(path_list[i])
        if cmd_args.encoder==True:
            graph_list=graph_encoder(graph_list,cfg)
        
        base_path = os.path.basename(path_list[i])
        name = os.path.splitext(base_path)[0]
        #assert len(graph_list) == cmd_args.n_graphs
        if i <num_train:
            train_glist += [(graph_list, name, sequence_list,path_list[i])]
        else:
            test_glist += [(graph_list, name, sequence_list,path_list[i])]
        #label_map[i] = i - cmd_args.min_c

    print('# train:', len(train_glist), ' # test:', len(test_glist))

    return  train_glist, test_glist

def load_graph_only(path_list):
    
    #pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)

    
    glist = []
    label_map = {}
    for i in range(len(path_list)):
        graph_list, sequence_list = generate_protein_features(path_list[i])
        
        base_path = os.path.basename(path_list[i])
        name = os.path.splitext(base_path)[0]
        glist += [(graph_list, name, sequence_list,path_list[i])]
        

    print('# total_g:', len(glist))

    return  glist

def test_graphs(classifier, test_glist):
    test_loss = loop_dataset(test_glist, classifier, list(range(len(test_glist))))
    print('\033[93maverage test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

def load_base_model(label_map, test_glist = None):
    assert cmd_args.base_model_dump is not None
    with open('%s-args.pkl' % cmd_args.base_model_dump, 'rb') as f:
        base_args = cp.load(f)

    classifier = GraphClassifier(label_map, **vars(base_args))
    if cmd_args.ctx == 'gpu':
        classifier = classifier.cuda()

    classifier.load_state_dict(torch.load(cmd_args.base_model_dump + '.model'))
    if test_glist is not None:
        test_graphs(classifier, test_glist)

    return classifier

def attackable(classifier, s2v_g, x = None, y = None):
    g = s2v_g.to_networkx()
    comps = [c for c in nx.connected_component_subgraphs(g)]
    set_id = {}

    for i in range(len(comps)):
        for j in comps[i].nodes():
            set_id[j] = i
    
    if x is not None:
        r_i = [x]
    else:
        r_i = range(len(g) - 1)

    g_list = []    
    for i in r_i:
        if y is not None:
            assert x is not None
            r_j = [y]
        else:
            if x is not None:
                r_j = range(len(g) - 1)
            else:
                r_j = range(i + 1, len(g))
        for j in r_j:
            if set_id[i] != set_id[j]:
                continue
            g2 = g.copy()
            g2.add_edge(i, j)
            assert nx.number_connected_components(g2) == s2v_g.label
            g_list.append(S2VGraph(g2, s2v_g.label))
    if len(g_list) == 0:
        print(x, y)
        print(g.edges(), s2v_g.label)
        print(set_id)
    assert len(g_list)
    _, _, acc = classifier(g_list)

    return np.sum(acc.view(-1).numpy()) < len(g_list)

def get_supervision(classifier, list_st, list_at):
    list_target = torch.zeros(len(list_st))
    for i in range(len(list_st)):
        g, x, _ = list_st[i]
        if x is not None:
            att = attackable(classifier, g, x, list_at[i])
        else:
            att = attackable(classifier, g, list_at[i])

        if att:            
            list_target[i] = 1.0
        else:
            list_target[i] = -1.0
    
    return list_target