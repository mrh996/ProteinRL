import numpy as np

def extract_pairs(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                line1 = lines[i].strip().split('\t')
                line2 = lines[i + 1].strip().split('\t')
                pairs.append(line1)
                pairs.append(line2)
    return pairs

def save_pairs_as_npy(pairs, output_file_path):
    np_array = np.array(pairs, dtype=object)
    np.save(output_file_path, np_array)

# Example usage
input_file_path = '/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/largescale.txt'  # replace with your actual input file path
output_file_path = 'large_data.npy'  # replace with your desired output file path

pairs = extract_pairs(input_file_path)
save_pairs_as_npy(pairs, output_file_path)

print(f"Pairs have been saved to {output_file_path}")