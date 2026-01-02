import numpy as np
from skopt import gp_minimize
import sys
import os
from skopt.space import Integer, Categorical
import glob
import csv
import time
from prediction_ddg import mutation_caller,predict_reward
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from rl_common import GraphEdgeEnv, local_args, load_graphs, load_graph_only,test_graphs, load_base_model, get_supervision
sys.path.append('%s/../../../PreMut/src' % os.path.dirname(os.path.realpath(__file__)))
from prediction_script import predictionClass
import random
random.seed(2015)
class BayesianOptimizationBaseline:
    def __init__(self, environment):
        """
        Initialize the Bayesian Optimization baseline.

        :param environment: Instance of the GraphEdgeEnv class or equivalent.
        """
        self.env = environment
        self.nodes = self.env.nodes[0]
        self.amino_acids = self.env.possible_table
        self.wild_name=self.env.wild_names
        self.pdb_path=self.env.wild_pdb_path
        self.protein_sequences=self.env.protein_sequences
        self.output_file = '/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/baye_10_large'

    def reward_function(self, params):
        """
        Reward function to optimize.

        :param params: [node_index, amino_acid_index]
        :return: Negative reward (since we minimize in Bayesian Optimization).
        """
        node_index, amino_acid_index = params
        node = node_index
        amino_acid = self.amino_acids[amino_acid_index]
        selected_node_name=self.protein_sequences[0][node]
        if str(amino_acid)==str(selected_node_name):
            reward=-10
        else:
            # Simulate a mutation and get the reward
            
            mutation_info = str(selected_node_name)+'_'+str(node)+'_'+str(amino_acid)
            reward,_,_ = predict_reward(self.pdb_path[0], self.wild_name[0], mutation_info, cfg = self.env.cfg)  # Using predict_reward from RL setup
            print(reward)

        return -reward  # Minimize negative reward

    def optimize(self, n_calls,start_time):
        """
        Perform Bayesian Optimization to select the best mutation.

        :param n_calls: Number of function evaluations.
        :return: Best node and amino acid type found.
        """
        # Define search spaces
        
        node_space = Integer(0, self.nodes - 1)
        amino_acid_space = Integer(0, len(self.amino_acids) - 1)

        # Run Bayesian Optimization
        res = gp_minimize(
            self.reward_function,
            dimensions=[node_space, amino_acid_space],
            n_calls=n_calls,
            random_state=10
        )

        best_node_index, best_amino_acid_index = res.x
        best_node =best_node_index
        best_amino_acid = self.amino_acids[best_amino_acid_index]
        selected_node_name=self.protein_sequences[0][best_node]
        
        mutation_info = str(selected_node_name)+'_'+str(best_node)+'_'+str(best_amino_acid)
        
        b_reward,_,_ = predict_reward(self.pdb_path[0], self.wild_name[0], mutation_info, cfg = self.env.cfg)  # Using predict_reward from RL setup
        
        # Save the best mutation info and reward to file
        end_time = time.time()
        times=end_time-start_time
        with open(self.output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.wild_name, times, mutation_info, b_reward])

        return best_node, best_amino_acid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skopt.space import Integer

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BayesianOptimizationENN:
    def __init__(self, environment):
        self.env = environment
        self.nodes = self.env.nodes[0]
        self.amino_acids = self.env.possible_table
        self.protein_sequences = self.env.protein_sequences
        self.pdb_path = self.env.wild_pdb_path
        self.wild_name = self.env.wild_names
        self.output_file = '/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/enn_10_large'
        self.ensemble_size = 5

        # Initialize ensemble of neural networks
        self.models = [SimpleNN(input_dim=2) for _ in range(self.ensemble_size)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in self.models]
        self.criterion = nn.MSELoss()

    def reward_function(self, params):
        node_index, amino_acid_index = params
        selected_node_name = self.protein_sequences[0][node_index]
        amino_acid = self.amino_acids[amino_acid_index]
        if str(amino_acid)==str(selected_node_name):
            reward=-10
        else:
            # Simulate a mutation and get the reward
            mutation_info = f"{selected_node_name}_{node_index}_{amino_acid}"
            reward,_,_ = predict_reward(self.pdb_path[0], self.wild_name[0], mutation_info, cfg = self.env.cfg)  # Using predict_reward from RL setup
            print(reward)
        
        return -reward

    def train_ensemble(self, X, y):
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for _ in range(50):  
                inputs = torch.tensor(X, dtype=torch.float32)
                targets = torch.tensor(y, dtype=torch.float32).view(-1, 1)
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = self.criterion(predictions, targets)
                loss.backward()
                optimizer.step()

    def predict_with_uncertainty(self, X):
        predictions = np.array([model(torch.tensor(X, dtype=torch.float32)).detach().numpy().flatten()
                                for model in self.models])
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred

    def optimize(self, n_calls, start_time):
        X = []
        y = []

        for _ in range(n_calls):
            if len(X) > 0:
                # Use acquisition function (e.g., UCB) to select the next point
                _, std_pred = self.predict_with_uncertainty(X)
                next_index = np.argmax(std_pred)
                params = X[next_index]
            else:
                # Random initialization for the first few points
                params = [np.random.randint(0, self.nodes), np.random.randint(0, len(self.amino_acids))]

            reward = self.reward_function(params)
            X.append(params)
            y.append(reward)

            self.train_ensemble(X, y)

        # Get the best point
        best_index = np.argmax(y)
        best_node, best_amino_acid_index = X[best_index]
        best_amino_acid = self.amino_acids[best_amino_acid_index]
        mutation_info = f"{self.protein_sequences[0][best_node]}_{best_node}_{best_amino_acid}"
        if str(self.protein_sequences[0][best_node])==str(best_amino_acid):
            b_reward=-10
        else:
            b_reward, _, _ = predict_reward(self.pdb_path[0], self.wild_name[0], mutation_info, cfg=self.env.cfg)

        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(self.output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.wild_name, elapsed_time, mutation_info, b_reward])

        return best_node, best_amino_acid
class RandomSearch:
    def __init__(self, environment):
        self.env = environment
        self.nodes = self.env.nodes[0]
        self.amino_acids = self.env.possible_table
        self.protein_sequences = self.env.protein_sequences
        self.pdb_path = self.env.wild_pdb_path
        self.wild_name = self.env.wild_names
        self.output_file = '/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/baye_random_small'

    def reward_function(self, params):
        node_index, amino_acid_index = params
        selected_node_name = self.protein_sequences[0][node_index]
        amino_acid = self.amino_acids[amino_acid_index]
        if str(amino_acid) == str(selected_node_name):
            return -10
        else:
            mutation_info = f"{selected_node_name}_{node_index}_{amino_acid}"
            reward, _, _ = predict_reward(self.pdb_path[0], self.wild_name[0], mutation_info, cfg=self.env.cfg)
            return -reward

    def optimize(self, n_calls, start_time):
        best_reward = -float('inf')
        best_params = None

        for _ in range(n_calls):
            params = [np.random.randint(0, self.nodes), np.random.randint(0, len(self.amino_acids))]
            reward = self.reward_function(params)
            if reward > best_reward:
                best_reward = reward
                best_params = params

        best_node, best_amino_acid_index = best_params
        best_amino_acid = self.amino_acids[best_amino_acid_index]
        mutation_info = f"{self.protein_sequences[0][best_node]}_{best_node}_{best_amino_acid}"

        b_reward, _, _ = predict_reward(self.pdb_path[0], self.wild_name[0], mutation_info, cfg=self.env.cfg)

        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(self.output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.wild_name, elapsed_time, mutation_info, b_reward])

        return best_node, best_amino_acid
# Example usage
if __name__ == "__main__":
    from rl_common import GraphEdgeEnv
    output_dir = cmd_args.save_dirsingle #_new
    prediction = predictionClass(save_dir=output_dir)
    # Assume env is an instance of GraphEdgeEnv or similar class
    directory = '/LOCAL2/mur/MRH/protein_RL/largescale_wt_fixed/' #largescale_wt_fixed large_test 
    graph_file_paths = glob.glob(os.path.join(directory, '*.pdb'))
    test_glist = load_graph_only(graph_file_paths)
    start_time = time.time()
    for idx, graph in enumerate(test_glist):
        start_p_time = time.time()
        env = GraphEdgeEnv(prediction)
        env.setup([graph])
        baseline = RandomSearch(env)
        best_node, best_amino_acid = baseline.optimize(10,start_p_time)
        
    print(f"Best node to mutate: {best_node}")
    print(f"Best amino acid to mutate to: {best_amino_acid}")