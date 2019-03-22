import numpy as np
import matplotlib.pyplot as plt
from mrp import machine_repair

env = machine_repair()
policy = np.zeros((env.nS)) + 1 #all not repair
policy[-6] = 0
num_trial = 3000
returns = np.zeros(num_trial)

for i in range(num_trial):
    o = env.reset()
    terminal = False
    ret = 0
    while not terminal:
        a = policy[o]
        no, r, terminal = env.step(a)
        ret += r
        o = no
    returns[i] = ret
    print(i)
np.save('results/reutrns_p7.npy', returns)

#from terrain import Nav2D
#env = Nav2D()
#for i in range(100):
#	env.step(np.random.randint(env.nA))
#	env._render()
