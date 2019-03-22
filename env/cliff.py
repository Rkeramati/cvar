import numpy as np
import matplotlib.pyplot as plt
# Defining a machine repair class, for machine repair env
# terminal State is a recuring state
class Cliff():
    def __init__(self, delta, M):
        #np.random.seed(seed)

        self.action_space = ["UP", "RI", "DO", "LE"]
        self.nA = len(self.action_space)

        self.maxX = 6
        self.maxY = 4
        self.nS = self.maxX * self.maxY

        self.gamma = 0.95
        self.delta = delta
        self.M = -M

        self.initial_state = (0, 0)
        self.goal_state = (self.maxX-1, 0)

        self.cliff = []
        for i in range(1, self.maxX-1):
            self.cliff.append((i, 0))
        self.terminal = False
        self.current_state = self.initial_state

    def idx(self, state):
        x, y = state
        return x * self.maxY + y

    def reset(self):
        self.current_state = self.initial_state
        self.terminal = False
        return self.idx(self.current_state)

    def step(self, action):
        if np.random.rand() <= self.delta:
            action = np.random.randint(self.nA)

        x, y =self.current_state
        if self.terminal:
            return self.idx(self.current_state), -1,  self.terminal

        reward = -1
        # Move
        if self.action_space[action] == "RI":
            x+=1
        if self.action_space[action] == "LE":
            x-=1
        if self.action_space[action] == "DO":
            y-=1
        if self.action_space[action] == "UP":
            y+=1

        #check Wall:
        hit = False
        if x>= self.maxX - 1:
            x = self.maxX - 1
            hit = True
        if x<=0:
            x = 0
            hit = True
        if y>=self.maxY - 1:
            y=self.maxY - 1
            hit = True
        if y<=0:
            y=0
            hit = True

        self.current_state = (x, y)
        #if hit:
            #self.terminal = True
            #return self.idx(self.current_state), reward, self.terminal

        # Check Obstacle
        if self.current_state in self.cliff:
            reward = -1 + self.M
            self.terminal = True

        #Check goal:
        if self.current_state == self.goal_state:
            reward = 0
            self.terminal = True

        return self.idx(self.current_state), reward, self.terminal

    def _render(self):
        p = np.zeros((self.maxX, self.maxY))
        for state in self.cliff:
            x, y = state
            p[x, y] = 1
        x, y = self.current_state
        p[x, y] = 2
        p[self.goal_state] = 3
        plt.matshow(p.T)
        plt.show()

