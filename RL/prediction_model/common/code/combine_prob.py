from __future__ import print_function
import os
import sys
import numpy as np
import torch
import networkx as nx
import random
import glob
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import torch
import csv
import random

random.seed(100)
print(torch.version.cuda)


sys.path.append('%s/../../../PreMut/src' % os.path.dirname(os.path.realpath(__file__)))
from prediction_script import predictionClass

from q_net import NStepQNet, QNet, greedy_actions
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

from rl_common import GraphEdgeEnv, local_args, load_graphs, load_graph_only,test_graphs, load_base_model, attackable, get_supervision
from nstep_replay_mem import NstepReplayMem

sys.path.append('%s/../graph_classification' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset

  
    
    

class Agent(object):
    def __init__(self, g_list, test_g_list, env):
        self.g_list = g_list
        #if test_g_list is None:
        self.test_g_list = test_g_list
        
        self.mem_pool = NstepReplayMem(memory_size=50000, n_steps=2)
        self.env = env
        # self.net = QNet()
        self.net = NStepQNet(2)
        self.old_net = NStepQNet(2)
        if cmd_args.ctx == 'gpu':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        self.eps_start = 1.0
        self.eps_end = 1.0
        self.eps_step = 20000
        self.burn_in = 10       
        self.step = 0

        self.best_eval = None
        self.pos = 0
        self.sample_idxes = list(range(len(g_list)))
        random.shuffle(self.sample_idxes)
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
            #print('random action is',actions)
        else:
            cur_state = self.env.getStateRef() # first node, ban list
            #print('cur_state is', list(cur_state))
            actions, _, _ = self.net(time_t, cur_state, None, greedy_acts=True)
            actions = list(actions.cpu().numpy())
            
        return actions

    def run_simulation(self):
        if (self.pos + 1) * cmd_args.batch_size > len(self.sample_idxes):
            self.pos = 0
            random.shuffle(self.sample_idxes)
        selected_idx = self.sample_idxes[self.pos * cmd_args.batch_size : (self.pos + 1) * cmd_args.batch_size]
        self.pos += 1
        self.env.setup([self.g_list[idx] for idx in selected_idx])

        t = 0
        while not env.isTerminal():
            
            list_at = self.make_actions(t)
            list_st = self.env.cloneState()
            self.env.step(list_at)

            #assert (env.rewards is not None) == env.isTerminal()
            if env.isTerminal():
                rewards = env.rewards
                s_prime = None
            else:
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()
            # Convert to lists if they are zips
            if isinstance(list_st, zip):
                list_st = list(list_st)
            if isinstance(list_at, zip):
                list_at = list(list_at)
            if isinstance(rewards, zip):
                rewards = list(rewards)
            if isinstance(s_prime, zip):
                s_prime = list(s_prime)

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t)
            t += 1

    def eval(self):
        self.env.setup(deepcopy(self.test_g_list))
        t = 0
        while not self.env.isTerminal():
            list_at = self.make_actions(t, greedy=True)
            self.env.step(list_at)
            t += 1
        #test_loss = loop_dataset(env.graph_list, env.classifier, list(range(len(env.graph_list))))
        #print('\033[93m average test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))
        reward = np.mean(self.env.rewards)
        if cmd_args.phase == 'train' and self.best_eval is None:
            print('----saving to best agent.----')
            torch.save(self.net.state_dict(), cmd_args.save_dirsingle + '/epoch-best_reward.model')
            with open(cmd_args.save_dirsingle + '/epoch-best.txt', 'w') as f:
                f.write('%.4f\n' % reward)
            self.best_eval = reward
        return reward

    def train(self):
        log_out = open(cmd_args.logfile, 'w')
        pbar = tqdm(range(self.burn_in), unit='batch')
        for p in pbar:
            self.run_simulation()
        pbar = tqdm(range(local_args.num_steps), unit='steps')
        optimizer = optim.Adam(self.net.parameters(), lr=cmd_args.learning_rate)
        for self.step in pbar:

            self.run_simulation()

            if self.step % 100== 0:
                self.take_snapshot()
            if self.step % 100 == 0:
                r = self.eval()
                log_out.write('%d %.6f' % (self.step, r))

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=cmd_args.batch_size)

            list_target = torch.Tensor(list_rt)
            if cmd_args.ctx == 'gpu':
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                _, _, banned = zip(*cleaned_sp)
                _, q_t_plus_1, prefix_sum_prime = self.old_net(cur_time + 1, cleaned_sp, None)
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
                list_target[nonterms] = q_rhs
            
            # list_target = get_supervision(self.env.classifier, list_st, list_at)
            list_target = Variable(list_target.view(-1, 1))

            _, q_sa, _ = self.net(cur_time, list_st, list_at)

            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss) )

        log_out.close()
if __name__ == '__main__':
    cmd_args.phase ='eva'
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    

    
    directory = '/LOCAL2/mur/MRH/protein_RL/large_ss' #mega_pdb_test/positivemutpdb, pdb_S669,pdb_ssym, largescale_wt_fixed
    graph_file_paths = glob.glob(os.path.join(directory, '*.pdb'))
    test_glist = load_graph_only(graph_file_paths) 
    
    
    output_dir = cmd_args.save_dirsingle
    prediction = predictionClass(save_dir=output_dir)
    env = GraphEdgeEnv(prediction)
    #if cmd_args.frac_meta > 0:
        #num_train = int( len(g_list) * (1 - cmd_args.frac_meta) )
        #agent = Agent(train_glist, test_glist, env)
    #else:
    
    

    if cmd_args.phase =='eva':
      print('====begin eval======')
      agent = Agent(test_glist,test_glist, env)
      #agent.net.load_state_dict(torch.load('/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/test_new' + '/epoch-best_reward.model'))
      agent.net.load_state_dict(torch.load(cmd_args.save_dirsingle + '/epoch-best_reward.model'))
      policy_net = agent.net 
      agent.eval()
      for idx in range(len(test_glist)):
        #selected_idx=[i]
        #env.setup([test_glist[idx] for idx in selected_idx])
        env.setup([test_glist[idx]])
        data1=['1bz6a']
        data2=['2ocja']
        
        #data_file = os.path.join('/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code', 'loss.csv')
        #with open(data_file, 'w', encoding='utf-8', newline='') as f:
         #   writer = csv.writer(f)
         #   writer.writerow(['name', 'action_1', 'action_2'])
        t=0
        
        joint_probs = None
        
        # Step 1: Choose position
        cur_state = env.getStateRef()
        actions, q_values, prefix_sum = policy_net(t, cur_state, None, greedy_acts=False)
        
        # Extract probabilities for positions (high-level policy)
        position_probs = F.softmax(q_values.squeeze(), dim=0).detach().cpu().numpy()
        # Save results to CSV
        output_dir = cmd_args.save_dirsingle+'/pdb_ss'  # pdb_mega, pdb_S669,ssym,large
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(graph_file_paths[idx])
        outpur_csv_dir=output_dir
        output_file = os.path.join(outpur_csv_dir, filename+'.csv')
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Position', 'Joint Probability'])
            
            for pos in range(position_probs.shape[0]):
                writer.writerow([pos, position_probs[pos]])
        
        '''
        print('position_probs shape is',position_probs.shape)
        # Step 2: Choose protein type for each position
        t = 1
        joint_probs = np.zeros((len(position_probs), 20))  # Assuming 20 protein types
        
        for pos in range(len(position_probs)):
            env.setup([test_glist[idx]])
            # Simulate choosing this position
            simulated_actions = torch.tensor([pos]).long()
            env.step(simulated_actions)
            cur_state=env.getStateRef() # first node, ban list
            if cmd_args.ctx == 'gpu':
                simulated_action = simulated_action.cuda()
            
            # Get protein type probabilities for this position
            _, q_values_protein, _ = policy_net(t, cur_state, None, greedy_acts=False)
            protein_probs = F.softmax(q_values_protein[:20].squeeze(), dim=0).detach().cpu().numpy()
            
            # Calculate joint probabilities for this position
            joint_probs[pos] = position_probs[pos] * protein_probs

        # Save results to CSV
        output_dir = cmd_args.save_dirsingle+'/pdb_large'  # pdb_mega, pdb_S669,ssym,large
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(graph_file_paths[idx])
        outpur_csv_dir=output_dir
        output_file = os.path.join(outpur_csv_dir, filename+'.csv')
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Position', 'Protein Type', 'Joint Probability'])
            
            for pos in range(joint_probs.shape[0]):
                for prot in range(joint_probs.shape[1]):
                    writer.writerow([pos, prot, joint_probs[pos, prot]])

        print(f"###########Results saved to {output_file}#############")

        # You can also get the top N combinations if needed
        flat_joint_probs = joint_probs.flatten()
        top_indices = np.argsort(flat_joint_probs)[::-1][:10]  # Top 10 combinations
        print("\nTop 10 position-protein combinations:")
        for idx in top_indices:
            pos, prot = np.unravel_index(idx, joint_probs.shape)
            print(f"Position {pos}, Protein Type {prot}: {joint_probs[pos, prot]:.6f}")
        '''

        
