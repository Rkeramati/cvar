import numpy as np

def scenario_fun(): #Returning a Custom Scenario
    # Time hour + start time
    # Meal CHO = Meal/3
    return [(1, 60), (3, 20), (5, 60)]
def minDiff(sample_time):
    arr = [x[0] * 60/sample_time for x in scenario_fun()]
    arr = sorted([0] + arr)

    # Initialize Result
    minDiff = 10**20
    n = len(arr)

    for i in range(n - 1):
        if (arr[i+1] - arr[i] < minDiff):
            minDiff = arr[i+1] - arr[i]
    return minDiff

def reward_fun(BG):
    if BG[-1] <= 39:
        return -10
    b = BG[-1]/18.018018
    if b < 6:
        return -(b - 6)**2/5
    else:
        return -(b - 6)**2/10

def done_fun(BG):
    # Never Terminate, untill the end of the simulation
    return False

def get_action(index, action_map):

    # get action from action_map
    return action_map[index, :]

def discounted_return(returns, gamma):
    ret = 0
    for r in reversed(returns):
        ret = r + gamma * ret
    return ret

def custom_step(env, action, step, max_step, delay):
    reward = []
    num_step = 0

    # Action Delay
    if delay > 0:
        delayed = 0; terminal = False
        while delayed <= delay and not terminal:
            obs, rew, terminal, info = env.step([0, 0])
            num_step += 1
            delayed += 1
    # Action
    obs, rew, terminal, info = env.step(action)
    num_step += 1
    reward.append(rew)
    meal = info['meal']
    # Till next meal
    while meal <= 0 and not terminal and step + num_step <= max_step:
        obs, rew, terminal, info = env.step([0, 0])
        meal = info['meal']
        num_step += 1
        reward.append(rew)
    return obs, np.mean(reward), terminal, info, num_step

