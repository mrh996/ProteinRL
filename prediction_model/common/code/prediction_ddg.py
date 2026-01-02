import torch
from torch_geometric.data import Data
import os
from configs_pre import get_cfg_defaults
from configs import get_cfg_defaults as get_model_cfg
from model_ddg import Main_model
import contactmap_features as cmf
import sys
# TO IMPORT PreMut MODEL TO GET MUTATION PDB FILE
sys.path.insert(0, "/LOCAL2/mur/MRH/protein_RL/PreMut/src")
from prediction_script import main

def load_model(model_path, model_config, device):
    """
    Load the ddg prediction model from a file.
    """
    model = Main_model(**model_config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def diff_generation(wt_image, mut_image):
    """
    :param wt_image: wild_type contact map graph
    :param mut_image: mutation contact map graph
    :return: diff_graph
    """
    wt_node_features = wt_image.x
    wt_edge_index = wt_image.edge_index

    mut_node_features = mut_image.x
    mut_edge_index = mut_image.edge_index

    # Calculate difference image
    node_diff = wt_node_features - mut_node_features
    encoded_edges_mut = mut_edge_index[0] * max(wt_edge_index.max(), mut_edge_index.max()) + mut_edge_index[1]
    encoded_edges_wt = wt_edge_index[0] * max(wt_edge_index.max(), mut_edge_index.max()) + wt_edge_index[1]
    unique_edges = torch.cat([encoded_edges_wt, encoded_edges_mut]).unique(return_counts=True)
    unique_edges_mask = unique_edges[1] == 1
    unique_encoded_edges = unique_edges[0][unique_edges_mask]
    unique_edge_index = torch.stack([unique_encoded_edges // max(wt_edge_index.max(), mut_edge_index.max()), unique_encoded_edges % max(wt_edge_index.max(), mut_edge_index.max())])
    graph_diff = Data(x=node_diff, edge_index=unique_edge_index, batch=wt_image.batch)

    return graph_diff

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
'''
def mutation_caller(wild_pdb_path, mutation_info, output_dir, therhold, rm):
    """
    Improved mutation caller with proper error handling and validation.
    
    Args:
        wild_pdb_path (str): Path to wild type PDB file
        mutation_info (str): Mutation information string
        output_dir (str): Directory to save mutation PDB file
        therhold (float): Threshold for the DDG prediction model
        rm (bool): Whether to remove temporary files
        
    Returns:
        tuple: (Graph_mut, mut_pdb_path) or (None, None) if error
    """
    try:
        # Validate inputs
        if not os.path.exists(wild_pdb_path):
            print(f"Error: Wild type PDB file not found: {wild_pdb_path}")
            return None, None
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        chain_id = 'A'
        
        # Generate mutation file name
        mut_file_name = wild_pdb_path.split('.')[0].split('/')[-1] + '_' + chain_id + '_' + mutation_info
        mut_pdb_path = os.path.join(output_dir, mut_file_name + '.pdb')
        
        print(f"\nProcessing mutation:")
        print(f"Wild PDB: {wild_pdb_path}")
        print(f"Mutation: {mutation_info}")
        print(f"Output path: {mut_pdb_path}")
        
        # Call main function from prediction script
        result = main(wild_pdb_path, mutation_info, output_dir, chain_id)
        if result is None:
            print("Error: Failed to generate mutation structure")
            return None, None
            
        # Verify mutation file was created
        if not os.path.exists(mut_pdb_path):
            print(f"Error: Mutation PDB file was not generated at {mut_pdb_path}")
            return None, None
            
        # Generate features
        try:
            Graph_mut, seq_mut = cmf.generate_protein_features(mut_pdb_path, mut_file_name, therhold)
        except Exception as e:
            print(f"Error generating protein features: {str(e)}")
            return None, None
            
        if Graph_mut is None:
            print("Error: Failed to generate mutation graph")
            return None, None
            
        # Clean up if requested
        if rm and os.path.exists(mut_pdb_path):
            try:
                os.remove(mut_pdb_path)
                print(f"Removed temporary file: {mut_pdb_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {str(e)}")
                
        return Graph_mut, mut_pdb_path
    except Exception as e:
        print(f"Error in mutation_caller: {str(e)}")
        return None, None
'''
def mutation_caller(wild_pdb_path, mutation_info, output_dir, therhold, rm):
    """
    :param wild_pdb_path: to get the pdb file
    :param mutation_info:
    :param output_dir: temprate folder to save mutation pdb file, if the cfg.PROTEIN.TEMP_RM is True, the mutation pdb file will be removed after using.
    :param therhold: save as the therehold of the ddg_prediction model.
    :return: mutation contact map graph
    """
    chain_id = 'A'
    # Call the main function from prediction_script
    main(wild_pdb_path, mutation_info, output_dir, chain_id)
    

    mut_file_name = wild_pdb_path.split('.')[0].split('/')[-1] #+ '_' + chain_id + '_' + mutation_info
    mut_pdb_path = output_dir + '/' + mut_file_name + '.pdb'
    
    #print(f"\nsave new graph to:",mut_pdb_path)
    #print(f"\nProcessing mutation:")
    #print(f"Wild PDB: {wild_pdb_path}")
    #print(f"Mutation: {mutation_info}")
    #print(f"Output path: {mut_pdb_path}")
    Graph_mut, seq_mut = cmf.generate_protein_features(mut_pdb_path, mut_file_name, therhold)
    #if rm:
        #os.remove(mut_pdb_path)

    return Graph_mut,mut_pdb_path
def main_predict_single(wt_graph_data, diff_graph_data, cfg):
    """
    :param wt_graph
    :param diff_graph
    :param cfg: parameters
    :return: predicted ddg (as the reward)
    """
    cfg_model = get_model_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model_path = os.path.join(cfg.DDG_MODEL.DIR, 'model')
    model = load_model(model_path, cfg_model, device)

    # Convert graph data to device and add batch dimension
    wt_graph_data = wt_graph_data.to(device)
    diff_graph_data = diff_graph_data.to(device)

    # Prediction
    with torch.no_grad():
        output, _ = model(wt_graph_data, diff_graph_data)
        prediction = output.cpu().numpy()

    return prediction
def predict_reward(wild_pdb_path, wild_name,mutation_info, cfg):
    print('wild_pdb_path is',wild_pdb_path,'mutation_info is',mutation_info)
    Graph_wt, seq_wt = cmf.generate_protein_features(wild_pdb_path, wild_name, cfg.PROTEIN.THERSHOLD)
    wt_graph_data = graph_to_data(Graph_wt)
    Graph_mut,mut_pdb_path = mutation_caller(wild_pdb_path, mutation_info, cfg.PROTEIN.TEMP, cfg.PROTEIN.THERSHOLD, cfg.PROTEIN.TEMP_RM)
    mut_graph_data = graph_to_data(Graph_mut)
    #print('wild_pdb_path is',wild_pdb_path)
    diff_graph_data = diff_generation(wt_graph_data, mut_graph_data)

    predict_ddg_list = main_predict_single(wt_graph_data, diff_graph_data, cfg)
    reward = float(predict_ddg_list[0])

    return reward,Graph_mut,mut_pdb_path

# Example usage:
if __name__ == '__main__':
    cfg = get_cfg_defaults()

    wild_name = '1poh'
    wild_pdb_path = '/LOCAL2/mur/MRH/protein_RL/largescale_wt_fixed/1poh.pdb'
    mutation_info = 'G_1_O'
    # mutation_info is what the agent need to generate

    therhold = cfg.PROTEIN.THERSHOLD
    temp_dir_pdb = cfg.PROTEIN.TEMP
    Graph_wt, seq_wt = cmf.generate_protein_features(wild_pdb_path, wild_name, therhold)
    wt_graph_data = graph_to_data(Graph_wt)

    rm = cfg.PROTEIN.TEMP_RM
    Graph_mut,_ = mutation_caller(wild_pdb_path, mutation_info, temp_dir_pdb, therhold, rm)
    mut_graph_data = graph_to_data(Graph_mut)
    diff_graph_data = diff_generation(wt_graph_data, mut_graph_data)

    predict_ddg_list = main_predict_single(wt_graph_data, diff_graph_data, cfg)
    predict_ddg = float(predict_ddg_list[0])
    print(predict_ddg)
    # predict_ddg is the reward