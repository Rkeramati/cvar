
def scenario_fun(): #Returning a Custom Scenario
    # Time hour + start time
    # Meal CHO = Meal/3
    return [(1, 60), (3, 20), (5, 60)]

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
