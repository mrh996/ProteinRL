import sys
# sys.path.insert(0,"./../PreMut/PreMut/src/")
sys.path.insert(0,"./../PreMut/PreMut/src")
from prediction_script import main
import pandas as pd
import os


file_path = './../../dataset/Largescale/largescale.txt'
mut_dir = './../../dataset/Largescale/mut/'
wt_dir = './../../dataset/Largescale/wt/'
out_dir ='./../../dataset/Largescale/mut/'

with open(file_path, 'r') as infile:
    for line in infile:
        fields = line.split('\t')
        pdb_file_name = fields[0] + '_A_' + fields[2] + '_' + fields[5] + '_' + fields[3] + '.pdb'
        mut_file = mut_dir + pdb_file_name
        wt_file = wt_dir + fields[0] + '.pdb'
        mutation_info = fields[2] + '_' + fields[5] + '_' + fields[3]

        if not os.path.exists(mut_file):
            chain_id = 'A'

            main(wt_file, mutation_info, out_dir, chain_id)
            print (pdb_file_name)
        else:
            print('exists')
