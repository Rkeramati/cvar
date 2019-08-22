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

        self.nAtoms = 151
        self.Vmin = -10
        self.Vmax = 40

        # Summary
        self.eval_episode = 100
        self.save_episode = 100
        self.print_episode = 1
        self.summary_write_episode = 100

        # Exploration
        sch = {1: (0.9, 0.05, 10), 2: (0.9, 0.05, 8), 3: (0.9, 0.05, 5), 4:(0.9, 0.05, 4), 5: (0.9, 0.05, 2), 6: (1.0, 0.9, 10), 7:(1.0, 0.9, 100), 8:(1.0, 0.9, 500), 9:(1, 0.99, 10), 10:(1.0, 0.99, 100)}
        if args.tune <=5:
            self.max_e, self.min_e, self.episode_ratio_e = sch[args.tune]
            self.linear = True
        else:
            self.linear = False
            self.max_e, self.decay, self.decay_step = sch[args.tune]

        self.max_lr = 0.001 # Maximum learnig rate
        self.min_lr = 0.0001
        self.episode_ratio = 2 # When to reach the minimum in episode for alpha and ep schedule

        #self.schedule = [0.9, 0.1, 2] # Epsilon greedy exploration scheduelce

        # Building Action Space

        self.nA = 4

        self.CVaRSamples = 50 # number of samples for CVaR

        self.train_size=32
        self.memory_size = 10000 # Size of the replay memory

        self.args = args
        self.action_map = np.array([0, 1, 2, 3])

    def get_lr(self, ep):
        slope = (-self.max_lr + self.min_lr)/(self.args.num_episode/self.episode_ratio)
        alpha = max(self.min_lr, self.max_lr + ep *slope)
        return alpha

    def get_epsilon(self, ep):
        if self.linear:
            slope = (-self.max_e + self.min_e)/(self.args.num_episode/self.episode_ratio_e)
            e = max(self.min_e, self.max_e + ep *slope)
        else:
            e = self.max_e * self.decay**(ep/self.decay_step)
        return e

    def action_process(self, a):
        action = np.zeros((a.shape[0], self.action_size))
        action[np.arange(a.shape[0]), a] = 1
        return action
