import numpy as np

class config():
    # Config file for C51 algorithm
    def __init__(self, env, args):
        # TF config
        architectures = {1: {1: 3, 2: [128, 128, 128]}, 2: {1: 4, 2:[128, 128, 128, 128]},\
                3: {1: 2, 2: [128, 128]}, 4: {1: 3, 2: [64, 64, 64]}}

        self.num_layers = architectures[args.arch][1] # Number of hidden layer
        self.layer_size = architectures[args.arch][2] # Hidden Layer size

        self.logprob_layers = [64, 64, 64] # RealNVP hidden layers

        self.state_size = 6
        self.action_size = 4
        # Log prob hyper:
        self.pg_epsilon = 0.000000001
        # Action space configuration

        self.nAtoms = 51
        self.Vmin = -1
        self.Vmax = 10

        # Summary
        self.eval_episode = 100
        self.save_episode = 100
        self.print_episode = 1
        self.summary_write_episode = 1e7

        # Exploration
        self.max_e = 0.9 # Exploration max epsilon
        self.min_e = 0.05
        self.max_lr = 0.001 # Maximum learnig rate
        self.min_lr = 0.0001
        self.episode_ratio = 2 # When to reach the minimum in episode for alpha and ep schedule

        #self.schedule = [0.9, 0.1, 2] # Epsilon greedy exploration scheduelce

        # Building Action Space

        self.nA = 4

        self.CVaRSamples = 20 # number of samples for CVaR

        self.train_size=32
        self.memory_size = 10000 # Size of the replay memory

        self.args = args

    def get_lr(self, ep):
        slope = (-self.max_lr + self.min_lr)/(self.args.num_episode/self.episode_ratio)
        alpha = max(self.min_lr, self.max_lr + ep *slope)
        return alpha

    def get_epsilon(self, ep):
        slope = (-self.max_e + self.min_e)/(self.args.num_episode/self.episode_ratio)
        e = max(self.min_e, self.max_e + ep *slope)
        return e