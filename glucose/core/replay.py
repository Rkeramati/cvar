import numpy as np
import pickle
''' Replay Buffer implemetation'''

class Replay():
    def __init__(self, config, load=False, name=None):
        print("Initialzing the replay buffer")
        self.size = config.memory_size

        if load:
            if name is None:
                raise Exception("name is not specifed for replay buffer load")
            self.load(name)
        else:
            self.actions = np.empty(self.size, dtype=np.int32)
            self.rewards = np.empty(self.size, dtype=np.float32)
            self.states = np.empty(self.size, dtype=np.int32)
            self.terminals = np.empty(self.size, dtype=np.bool)

            self.count = 0
            self.current = 0
    def add(self, state, action, reward, terminal):
         '''Add a new example to the replay memory'''
         self.actions[self.current] = action
         self.rewards[self.current] = reward
         self.states[self.current] = state
         self.terminals[self.current] = terminal

         self.count = max(self.count, self.current + 1)
         self.current = (self.current + 1) % self.size # Pointer to the current state
    def sample(self, size):
        ''' sample a batch of random observation'''
        index = np.random.randint(low=0, high=self.count-1, size=size) # Random Index

        action = self.actions[index]
        state = self.states[index]
        reward = self.rewards[index]
        terminal = self.terminals[index]
        next_state = self.states[index + 1]

        # Terminal state doesn't have a next state, DRL takes care of this

        return state, action, reward, next_state, terminal
    def save(self, name):
        save = {'actions': self.actions, 'rewards': self.rewards,\
                'states':self.states, 'terminals': self.terminals,\
                'count': self.count, 'current':self.current}
        pickle.dump(save, open(name + '_replay.p', "wb"))
        print("Replay buffer saved, name:" + name + ".p")

    def load(self, name):
        load = pickle.load(open(name + '_replay.p', 'rb'))
        self.actions = load['actions']
        self.rewards = load['rewards']
        self.states = load['states']
        self.terminals = load['terminals']
        self.count = load['counts']; self.current = load['current']

        print("Replay buffer load")

