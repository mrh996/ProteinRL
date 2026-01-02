from Bio import PDB
import numpy as np
import torch
from torch_geometric import data as DATA
from Bio.SeqUtils import seq1

# Load the PDB file for the structure
def load_pdb(pdb_file, pname):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pname, pdb_file)
    model = structure[0]

    seq = list(structure.get_residues())
    # Initialize an empty string to store the protein sequence
    protein_sequence = ''

    # Iterate through each residue in the sequence list
    for residue in seq:
        # Check if the residue is a standard amino acid to avoid heteroatoms like water molecules (HOH)
        if PDB.is_aa(residue, standard=True):
            # Get the three-letter code of the current residue
            three_letter_code = residue.get_resname()
            # Convert the three-letter code to a one-letter code and append it to the protein sequence string
            protein_sequence += seq1(three_letter_code)
    return structure, protein_sequence

def protein_features(candidate_file, pname, threshold):
    #print('candidate_file is',candidate_file,'pname',pname)
    candidate_structure, target_sequence = load_pdb(candidate_file, pname)
    g_size, g_features, g_edge_index, g_edge_attr = protein_global(pname, candidate_structure, target_sequence, threshold)
    return g_size, g_features, g_edge_index, g_edge_attr, target_sequence


def contact_map_generate(threshold, structure):
    num_residues = len(list(structure.get_residues()))
    contact_map = np.zeros((num_residues, num_residues), dtype=int)

    residue_ids = []
    ca_atoms_by_residue = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    residue_id = residue.get_id()
                    residue_ids.append(residue_id)
                    ca_atom = residue["CA"]
                    ca_atoms_by_residue[residue_id] = ca_atom.get_coord()

    for i, residue_i in enumerate(residue_ids):
        for j, residue_j in enumerate(residue_ids[i + 1:], start=i + 1):
            distance = np.linalg.norm(ca_atoms_by_residue[residue_i] - ca_atoms_by_residue[residue_j])
            if distance <= threshold:
                contact_map[i, j] = contact_map[j, i] = 1


    return contact_map

def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

def residue_features(residue):

    pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
    pro_res_aromatic_table = ['F', 'W', 'Y']
    pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
    pro_res_acidic_charged_table = ['D', 'E']
    pro_res_basic_charged_table = ['H', 'K', 'R']

    res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                        'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                        'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

    res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                     'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                     'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

    res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                     'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                     'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

    res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                     'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                     'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

    res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                    'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                    'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}


    res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                                 'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                                 'T': 13, 'V': 79, 'W': 84, 'Y': 49}

    res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                                 'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                                 'T': 13, 'V': 76, 'W': 97, 'Y': 63}

    res_weight_table = dic_normalize(res_weight_table)
    res_pka_table = dic_normalize(res_pka_table)
    res_pkb_table = dic_normalize(res_pkb_table)
    res_pkx_table = dic_normalize(res_pkx_table)
    res_pl_table = dic_normalize(res_pl_table)
    res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
    res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    res_property = res_property1 + res_property2
    return np.array(res_property)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def global_node_feature(pro_seq):
    pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                     'X']
    secondary_structures = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'P', '-']
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):

        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])

    feature = np.concatenate((pro_hot, pro_property), axis=1)

    return feature


def protein_global(target_key, target_structure, target_sequence, threshold):
    target_edge_index = []
    target_edge_attr = []  # Edge features, including contact and hydrogen bond information
    target_size = len(target_sequence)
    contact_map = contact_map_generate(threshold, target_structure)

    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
        target_edge_attr.append(contact_map[i, j])

    node_feature = global_node_feature(target_sequence)
    target_edge_index = np.array(target_edge_index)
    target_edge_attr = np.array(target_edge_attr)

    return target_size, node_feature, target_edge_index, target_edge_attr


def generate_protein_features(file, pname='a', threshold=8):
    #protein_graph, sequence='1','2'
    #print('file',file)
    
    g_size, g_features, g_edge_index, g_edge_attr, sequence = protein_features(file, pname, threshold)
    protein_graph = DATA.Data(x=torch.Tensor(g_features),
                              edge_index=torch.LongTensor(g_edge_index).transpose(1, 0),
                              edge_attr=torch.Tensor(g_edge_attr))  # Include edge attributes
    return protein_graph, sequence