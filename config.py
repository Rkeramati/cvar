'''
    Config Class
'''

class Config():
    def __init__(self, nS, nA):
        self.Vmin = -50 # Min Value for C51
        self.Vmax = 50
        self.nAtoms = 51 # nAtoms for C51
        self.nS = nS
        self.nA = nA
        self.gamma = 0.99 # Discount Factor
        self.max_e = 0.9 # Exploration max epsilon
        self.min_e = 0.1
        self.max_lr = 0.9 # Maximum learnig rate
        self.min_lr = 0.1
        self.episode_ratio = 4 # When to reach the minimum in episode for alpha and ep schedule
        self.CVaRSamples = 20 # Number of samples to compute CVaR
        self.eval = False # If train a eval network as well
        self.eval_trial = 10 # number of trial for evaluation/ either online or eval
        self.eval_episode = 20 # Evaluation episodes
        self.save_episode = 100 # Save episode
        self.e_greedy_eval = True # Evaluation for e-greedy

    def set(self, args):
        # Set the input configs
        self.args = args # for input values

    def get_lr(self, ep):
        slope = (-self.max_lr + self.min_lr)/(self.args.num_episode/self.episode_ratio)
        alpha = max(self.min_lr, self.max_lr + ep *slope)
        return alpha
    def get_epsilon(self, ep):
          slope = (-self.max_e + self.min_e)/(self.args.num_episode/self.episode_ratio)
          e = max(self.min_e, self.max_e + ep *slope)
          return e

