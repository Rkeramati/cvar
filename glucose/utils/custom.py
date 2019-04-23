
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
