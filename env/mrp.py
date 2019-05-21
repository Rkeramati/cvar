import numpy as np
# Defining a machine repair class, for machine repair env
# terminal State is a recuring state
class machine_repair():
    def __init__(self, uniform=False):
        self.nS = 25
        self.nA = 2
        self.maxRew = 23
        self.minRew = 10
        self.worseRew = 8
        self.scale = (self.maxRew - self.minRew)/self.nS

        self.uniform = uniform
        self.reset()
        self.final_state = self.nS - 2
        self.terminal_state = self.nS - 1
    def reset(self):
        if self.uniform:
            self.state = np.random.randint(self.nS)
        else:
            self.state = 0
        return self.state
    def step(self, action):
        if self.state == self.terminal_state:
            return self.state, 0, True

        terminal = False
        if self.state == self.final_state:
            if action == 1:#not-repair
                reward = -np.random.normal(self.worseRew, 10)
                terminal = True
                self.state = self.terminal_state
            elif action == 0:
                reward = -np.random.normal(self.maxRew - self.scale*self.state, 0.1+0.01*self.state)
                terminal = True
                self.state = self.terminal_state
            else:
                raise Exception("undefined action")

        else:
            if action == 1:
                reward = -np.random.normal(0, 1e-2 + 0.001*self.state)
                self.state += 1
            elif action == 0:
                reward = -np.random.normal(self.maxRew - self.scale*self.state, 0.1+0.01*self.state)
                terminal = True
                self.state = self.terminal_state
            else:
                raise Exception("undefined action")
        return self.state, reward, terminal
