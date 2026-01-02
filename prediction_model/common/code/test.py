import numpy as np
import pandas as pd
data = pd.read_csv('/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/5_clean_name.txt', sep=' ', header=None)
data_npy = data.values
np.save('large_data.npy', data_npy)