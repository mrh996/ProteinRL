from configs import get_cfg_defaults
import contactmap_features as cmf
import torch
from torch_geometric.data import Data
import pickle
import os


def check_graph(graph,pname):
    num_nodes = graph.x.shape[0]  
    edge_indices = graph.edge_index
    if edge_indices.numel() == 0:
        return
    if torch.max(edge_indices).item() >= num_nodes:
        print(
            f"Error in {pname}: edge_index contains more than node_index.")

    elif torch.min(edge_indices).item() < 0:
        print(
            f"Error in {pname}: edge_index contains less than node_index.")


def diff_graph_generation(wt_image,mut_image):
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
    unique_edge_index = torch.stack([unique_encoded_edges // max(wt_edge_index.max(), mut_edge_index.max()),
                                     unique_encoded_edges % max(wt_edge_index.max(), mut_edge_index.max())])
    graph_diff = Data(x=node_diff, edge_index=unique_edge_index)
    graph_wt = Data(x=wt_node_features, edge_index=wt_edge_index, edge_attr=wt_edge_attr)

    return graph_wt, graph_diff


def check_edge_index(edge_index, num_nodes, protein_name):
    if edge_index.max() >= num_nodes:
        print(
            f"Error in {protein_name}: edge_index contains invalid node index {edge_index.max().item()}, but there are only {num_nodes} nodes.")
    if edge_index.min() < 0:
        print(f"Error in {protein_name}: edge_index contains negative node index.")

def compare_sequences(seq_wt: str, seq_mut: str) -> list:
    
    diff_positions = [i for i, (wt, mut) in enumerate(zip(seq_wt, seq_mut)) if wt != mut]
    return diff_positions

def graph_to_data(graph):
    x = graph.x.clone().detach().to(torch.float)
    edge_index = graph.edge_index.clone().detach().to(torch.long)
    edge_attr = graph.edge_attr.clone().detach().to(torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



if __name__ == "__main__":

    mut_dir = './mut/'
    wt_dir = './largescale_wt'
    save_dir = './pickle/'
    cfg = get_cfg_defaults()
    therhold = cfg["PROTEIN"]["THERSHOLD"]
    graph_wt_list = []
    graph_diff_list = []
    mut_pname_list = []
    ddg_list = []
    file_path = './largescale.txt'

    protein_images = {}

    with open(file_path, 'r') as infile:
        for line in infile:
            fields = line.split('\t')
            pdb_file_name = fields[0] + '_A_' + fields[2] + '_' + fields[5] + '_' + fields[3] + '.pdb'
            mut_file = mut_dir + pdb_file_name
            wt_file = wt_dir + fields[0] + '.pdb'

            mut_pname = fields[0] + '_A_' + fields[2] + '_' + fields[5] + '_' + fields[3]

            wt_graph_file = save_dir + fields[0] + '.pkl'
            mut_graph_file = save_dir + mut_pname + '.pkl'

            if not os.path.exists(wt_graph_file):
                with open(wt_graph_file, 'wb') as wt_f:
                    Graph_wt, seq_wt = cmf.generate_protein_features(wt_file, fields[0], therhold)
                    pickle.dump(Graph_wt, wt_f)
                    check_edge_index(Graph_wt.edge_index, Graph_wt.x.size(0), fields[0])
                    protein_images[fields[0]] = graph_to_data(Graph_wt)
                    print(f"File {wt_graph_file} done.")
            else:
                if fields[0] not in protein_images:
                    print(f"Load File {wt_graph_file}.")
                    Graph_wt = pickle.load(open(os.path.join(save_dir, wt_graph_file), 'rb'))
                    protein_images[fields[0]] = graph_to_data(Graph_wt)
                else:
                    print(f"Item exists.")

            if not os.path.exists(mut_graph_file):
                with open(mut_graph_file, 'wb') as mut_f:
                    Graph_mut, seq_mut = cmf.generate_protein_features(mut_file, mut_pname, therhold)
                    pickle.dump(Graph_mut, mut_f)
                    check_edge_index(Graph_mut.edge_index, Graph_mut.x.size(0), mut_pname)
                    protein_images[mut_pname] = graph_to_data(Graph_mut)
                    print(f"File {mut_graph_file} done.")
            else:
                if mut_pname not in protein_images:
                    print(f"Load File {mut_graph_file}.")
                    Graph_mut = pickle.load(open(os.path.join(save_dir, mut_graph_file), 'rb'))
                    protein_images[mut_pname] = graph_to_data(Graph_mut)
                else:
                    print(f"Item exists.")

            graph_wt, graph_diff = diff_graph_generation(Graph_wt, Graph_mut)

            check_graph(graph_diff, mut_pname)
            check_graph(Graph_wt, fields[0])

    with open('./large_dictionary.pickle', 'wb') as dicfile:
        pickle.dump(protein_images, dicfile)


