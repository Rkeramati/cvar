import numpy as np

class config():
    # Config file for C51 algorithm
    def __init__(self, env):
        self.max_basal = env.action_space.high[1] # max of basal
        self.min_basal = env.action_space.low[1] # min of basal
        self.max_bolus = env.action_space.high[0] # max of bolus
        self.min_bolus = env.action_space.low[0] # min of bolus

        self.basal_bin = 1
        self.bolus_bin = 5

        self.state_bin = 20
        self.max_state = 400
        self.max_meal = 80
        self.meal_bin = 10
        
        self.nAtoms = 51
        self.Vmin = -50
        self.Vmax = 50
        self.gamma = 0.95

        self.max_step = 500
        self.eval_episode = 50
        self.save_episode = 200
        self.num_episode = 10000
        self.print_episode = 10

        self.max_e = 0.9 # Exploration max epsilon
        self.min_e = 0.1
        self.max_lr = 0.5 # Maximum learnig rate
        self.min_lr = 0.1
        self.episode_ratio = 4 # When to reach the minimum in episode for alpha and ep schedule
 
        self.schedule = [0.9, 0.1, 4] # Epsilon greedy exploration scheduelce

        self.nS = self.state_bin * self.meal_bin
        self.bin_size = (self.max_state - 0)/self.state_bin
        self.mea_size = (self.max_meal - 0)/self.meal_bin

        self.bolus_map = np.linspace(self.min_bolus, self.max_bolus, self.bolus_bin)
        self.basal_map = np.linspace(self.min_basal, self.max_basal, self.basal_bin)
        self.action_map = np.transpose([np.tile(self.bolus_map, len(self.basal_map)), np.repeat(self.basal_map, len(self.bolus_map))])
        self.nA = self.action_map.shape[0]

    def get_lr(self, ep):
        slope = (-self.max_lr + self.min_lr)/(self.num_episode/self.episode_ratio)
        alpha = max(self.min_lr, self.max_lr + ep *slope)
        return alpha

    def get_epsilon(self, ep):
        slope = (-self.max_e + self.min_e)/(self.num_episode/self.episode_ratio)
        e = max(self.min_e, self.max_e + ep *slope)
        return e

    def process(self, state, meal):
        state = min(int(state.CGM/self.bin_size), self.state_bin)
        meal = min(int(meal/self.mea_size), self.meal_bin)
        idx = state + meal * self.meal_bin #state idx given mean and state
        return idx    