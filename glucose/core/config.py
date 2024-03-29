import numpy as np

class config():
    # Config file for C51 algorithm
    def __init__(self, env, args):
        print("Warning: Max Bolus/4")
        self.max_basal = env.action_space.high[1] # max of basal
        self.min_basal = env.action_space.low[1] # min of basal
        self.max_bolus = env.action_space.high[0]/4 # max of bolus
        self.min_bolus = env.action_space.low[0] # min of bolus

        self.basal_bin = 1
        self.bolus_bin = 5

        self.state_bin = 50
        self.max_state = 500
        self.max_meal = 60
        self.meal_bin = 3

        self.nAtoms = 51
        self.Vmin = -40
        self.Vmax = 15
        self.power_law = 1

        self.eval_episode = 10
        self.save_episode = 100
        self.print_episode = 1

        #self.max_e = 0.9 # Exploration max epsilon
        #self.min_e = 0.1
        self.max_lr = 0.9 # Maximum learnig rate
        self.min_lr = 0.5
        #self.episode_ratio = 2 # When to reach the minimum in episode for alpha and ep schedule

        self.schedule = {1: [0.9, 0.1, 2], 2: [0.9, 0.05, 4],\
                3: [0.9, 0.05, 6], 4: [0.9, 0.3, 4]} # Epsilon greedy exploration scheduelce

        self.max_e, self.min_e, self.episode_ratio = self.schedule[args.e_greedy_option]

        # self.exp_decay = {1: [0.9, 0.99, 5], 2:[0.9, 0.99, 20], 3:[0.9, 0.99, 2],\
        #                 4: [0.9, 0.99, 30], 5:[0.5, 0.99, 5]}
        # self.start_e, self.decay, self.decay_step = self.exp_decay[args.e_greedy_option]

        self.nS = self.state_bin * self.meal_bin
        self.bin_size = (self.max_state - 0)/self.state_bin
        self.meal_size = (self.max_meal - 0)/self.meal_bin

        self.bolus_map = np.linspace(self.min_bolus, self.max_bolus, self.bolus_bin)
        self.basal_map = np.linspace(self.min_basal, self.max_basal, self.basal_bin)
        self.action_map = np.transpose([np.tile(self.bolus_map, len(self.basal_map)), np.repeat(self.basal_map, len(self.bolus_map))])
        self.nA = self.action_map.shape[0]

        self.CVaRSamples = 20

        self.train_size=32
        self.memory_size = 10000

        self.args = args

    def get_lr(self, ep):
        slope = (-self.max_lr + self.min_lr)/(self.args.num_episode/self.episode_ratio)
        alpha = max(self.min_lr, self.max_lr + ep *slope)
        return alpha

    def get_epsilon(self, ep, decay=False):
        # if not decay:
        slope = (-self.max_e + self.min_e)/(self.args.num_episode/self.episode_ratio)
        e = max(self.min_e, self.max_e + ep *slope)
        # else:
            #e = self.start_e * self.decay**(ep/self.decay_step)
        return e

    def get_action(self, action_id):
        total_action = [0, 0]
        action = self.action_map[action_id, :]
        total_action[0] = action[0] #+ np.random.normal(0, self.args.action_sigma)
        return total_action

    def process(self, state, meal):

        state = min(int(state.CGM/self.bin_size), self.state_bin)
        # Randomize the state observation
        # if np.random.rand() <= self.args.delta_state:
        #     if np.random.rand() <= 0.5:
        #         state = max(0, state - 1)
        #     else:
        #         state = min(self.state_bin, state + 1)
        meal = min(int(meal/self.meal_size), self.meal_bin)
        idx = state + meal * self.meal_bin #state idx given mean and state
        return idx

    def get_delay(self):
        # Return the action delay
        step = self.args.action_delay - int(np.random.power(self.power_law) \
                * self.args.action_delay) - 1
        return max(0, step)

