
Code for paper ThermoRL:Structure-Aware Reinforcement Learning for Protein Mutation Design to Enhance Thermostability
# ThermoRL

# Train the surrogate model

In the surrogate file, set the environment from environment.yml

cd surrogate/surrogate/preprocess

python mut_generator.py

python picklegraph_generator.py

cd ../model

python k_fold_train.py

# Train the RL agent


cd RL/prediction_model

Save the trained surrogate model in the specific file 

cd common/code

python dqn.py 

You can set the save model file adress and whether use encoder in [text](RL/prediction_model/common/cmd_args.py)
