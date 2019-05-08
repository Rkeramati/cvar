import numpy as np

class config():
    # Config file for C51 algorithm
    def __init__(self, env, args):
        print("Warning: Max Bolus/4")
        # TF config
        self.num_layers = 2 # Number of hidden layer
        self.layer_size = [32, 32] # Hidden Layer size
        self.logprob_layers = [64, 64, 64] # RealNVP hidden layers
        self.state_size = 2
        self.action_size = 2
        # Log prob hyper:
        self.pg_epsilon = 0.000000001
        # Action space configuration
        self.max_basal = env.action_space.high[1] # max of basal
        self.min_basal = env.action_space.low[1] # min of basal
        self.max_bolus = env.action_space.high[0]/4 # max of bolus
        self.min_bolus = env.action_space.low[0] # min of bolus

        self.basal_bin = 1
        self.bolus_bin = 5
        self.power_law = 1

        # Normalizing state space
        self.max_state = 800
        self.min_state = 0
        self.max_meal = 60

        # C51
        self.nAtoms = 51
        self.Vmin = -40
        self.Vmax = 15

        # Summary
        self.eval_episode = 5
        self.save_episode = 10
        self.print_episode = 1
        self.summary_write_episode = 1

        # Exploration
        self.max_e = 0.9 # Exploration max epsilon
        self.min_e = 0.1
        self.max_lr = 0.9 # Maximum learnig rate
        self.min_lr = 0.5
        self.episode_ratio = 2 # When to reach the minimum in episode for alpha and ep schedule

        #self.schedule = [0.9, 0.1, 2] # Epsilon greedy exploration scheduelce

        # Building Action Space
        self.bolus_map = np.linspace(self.min_bolus, self.max_bolus, self.bolus_bin)
        self.basal_map = np.linspace(self.min_basal, self.max_basal, self.basal_bin)
        self.action_map = np.transpose([np.tile(self.bolus_map, len(self.basal_map)), np.repeat(self.basal_map, len(self.bolus_map))])
        self.nA = self.action_map.shape[0]

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

    def get_action(self, action_id):
        total_action = [0, 0]
        action = self.action_map[action_id, :]
        total_action[0] = action[0] + np.random.normal(0, self.args.action_sigma)
        return total_action

    def process(self, state, meal):
        out = np.zeros(self.state_size)
        out[0] = (state.CGM - self.max_state/2)/(self.max_state) # between -0.5, 0.5 for CGM
        out[1] = meal/(2 * self.max_meal) # Between 0 - 0.5 for meal amount

        if abs(out[0]) > 0.5:
            print("warning: State processing produced CGM: %g"%(out[0]))
        if abs(out[1]) > 0.5:
            print("warning: State processing produced Meal: %g"%(out[1]))

        return out

    def get_delay(self):
        # Return the action delay
        step = self.args.action_delay - int(np.random.power(self.power_law) \
                * self.args.action_delay) - 1
        return max(0, step)

    def action_process(self, a):
        max_action = np.array([[self.max_bolus, self.max_basal]])
        a = (a - max_action/2)/max_action
        return a
