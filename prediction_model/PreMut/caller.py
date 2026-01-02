import sys
#sys.path.insert(0,"src/")
from src.prediction_script import main


wild_pdb_path = '/LOCAL2/mur/MRH/protein_RL/PreMut/1msi.pdb'
mutation_info = 'R_47_A'
output_dir = '/LOCAL2/mur/MRH/protein_RL/PreMut/'
chain_id = 'A'
# Call the main function from prediction_script
main(wild_pdb_path,mutation_info,output_dir,chain_id)


