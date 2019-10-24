'''
    Config Class
'''

class Config():
    def __init__(self, nS, nA):
        self.nS = nS # State dimension 
        self.nA = nA # Action dimension

        self.num_tau = 8 # number of samples from base distribution
        self.num_tau_prime = 8 # number of samples for next state estimate

        self.max_lr = 1e-3 # maximum learning rate
        self.min_lr = 1e-5 # minimum learning rate
        self.max_e = 0.9 # Exploration max epsilon
        self.min_e = 0.05
        self.episode_ratio = 4 # When to reach the minimum in episode for alpha and ep schedule

        self.gamma = 0.9 # Discount Factor
        self.target_update_step = 25
        self.memory_size = 10000
        self.batch_size = 32
        self.gradient_norm = None # If normalize the gradeint

        self.eta = 0.25 # CVaR value = self.alpha
        
        self.CVaRSamples = 30 # Number of samples to compute CVaR
        self.eval = False # If train a eval network as well
        self.eval_trial = 20 # number of trial for evaluation/ either online or eval
        self.eval_episode = 50 # Evaluation episodes
        self.save_episode = 200 # Save episode
        self.print_episode = 10
        self.e_greedy_eval = False # not-implemented # Evaluation for e-greedy
        self.schedule = {1: [0.9, 0.1, 4], 2: [0.9, 0.05, 5], 3: [0.9, 0.3, 4],\
                4: [0.9, 0.05, 2], 5:[0.9, 0.05, 10], 6:[0.9, 0.05, 20], 7:[0.9, 0.05, 15],\
		8:[0.9, 0.1, 10]}

    def set(self, args):
        # Set the input configs
        self.args = args # for input values
        self.eta = args.alpha # alpha added as an options
        self.gamma = args.gamma # Gamma added as an option
        self.max_e, self.min_e, self.episode_ratio = self.schedule[self.args.option]
    def get_lr(self, ep):
        slope = (-self.max_lr + self.min_lr)/(self.args.num_episode/self.episode_ratio)
        alpha = max(self.min_lr, self.max_lr + ep *slope)
        return alpha
    def get_epsilon(self, ep):
          slope = (-self.max_e + self.min_e)/(self.args.num_episode/self.episode_ratio)
          e = max(self.min_e, self.max_e + ep *slope)
          return e

