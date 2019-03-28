import numpy as np
import matplotlib.pyplot as plt

# Navigation 2D task
class Nav2D():
    def __init__(self, random=False):
        # Setting the random seed
        #np.random.seed(seed)

        # Action Space
        self.action_space = ["UP", "RI", "DO", "LE"]
        self.nA = len(self.action_space)
        # Width and Heigth of the map
        self.maxX = 16
        self.maxY = 13
        self.nS = self.maxX * self.maxY #Number of states

        # Range of obstacles
        self.xRange = (0, 14)
        self.yRange = (2, 11)
        self.numObstacles = 15

        self.gamma = 0.95
        self.delta = 0.3
        self.M = 2/(1-self.gamma)

        # Initial and Goal state
        self.initial_state = (14, 11)
        self.goal_state = (14, 2)

        if random:
            self.obstacles = self.random_obstacles()
        else:
            self.obstacles = self.handcoded_obstacles()

        self.terminal = False
        self.current_state = self.initial_state

        self.counter = 0
        self.max_step = 1000 #maximum number of steps in each run

    def random_obstacles(self):
        # generate random obstacles
        obstacles = np.zeros((self.maxX, self.maxY))

        exp = 0.25
        p = np.zeros((self.maxX, self.maxY))
        for i in range(min(self.xRange), max(self.xRange)):
            px =  np.exp((i/max(self.xRange))**(exp))/np.exp(1) * 0.3
            for j in range(min(self.yRange), max(self.yRange)):
                py = 1-np.exp((abs((j-2) - 5)/5)**(5))/np.exp(1) * 0.3
                p[i,j] = px * py * np.random.randn()
        idx = np.argsort(-p.flatten())[0:self.numObstacles]

        prob = np.zeros((self.maxX, self.maxY)).flatten()
        prob[idx] = 1
        obstacles = prob.reshape(((self.maxX, self.maxY)))

        obstacles[self.initial_state] = 0
        obstacles[self.goal_state] = 0

        return obstacles

    def handcoded_obstacles(self):
        # Makes handcoded obstacles and return then matrix:
        obstacles = np.zeros((self.maxX, self.maxY))

        obstacles[13, 7] = 1
        obstacles[13, 6] = 1
        obstacles[13, 5] = 1
        obstacles[13, 9] = 1
        obstacles[13, 8] = 1
        obstacles[11, 4] = 1
        obstacles[11, 3] = 1
        obstacles[11, 8] = 1
        obstacles[8, 9] = 1
        obstacles[8, 7] = 1

        obstacles[7, 5] = 1
        obstacles[7, 8] = 1
        obstacles[5, 6] = 1

        obstacles[5, 4] = 1

        return obstacles

    def idx(self, state):
        x, y = state
        return x * self.maxY + y

    def reset(self):
        self.current_state = self.initial_state
        self.terminal = False
        self.counter = 0
        return self.idx(self.current_state)

    def step(self, action):
        if np.random.rand() <= self.delta:
            action = np.random.randint(self.nA)
        if self.counter >= self.max_step:
            self.terminal = True
        else:
            self.counter += 1

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
        #    reward += -self.M
        #    self.terminal = True
        #    return self.idx(self.current_state), reward, self.terminal
        # Checl Obstacle
        if self.obstacles[x, y] == 1:
            reward += -self.M
            self.terminal = True

        #Check goal:
        if self.current_state == self.goal_state:
            reward = 0
            self.terminal = True

        return self.idx(self.current_state), reward, self.terminal

    def _render(self):
        p = self.obstacles
        x, y = self.current_state
        p[x, y] = 2
        p[self.goal_state] = 3
        plt.matshow(p.T)
        plt.show()

