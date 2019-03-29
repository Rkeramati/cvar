import numpy as np

def eval(world, c51, trial, config):
    # Policy Evaluation with c51, CVaR optimization
    returns = np.zeros(trial)
    for ep in range(trial):
        terminal = False
        o = world.reset()
        ret = []
        while not terminal:
            values = c51.CVaR(o, alpha=config.args.alpha, N=config.CVaRSamples)
            a = np.random.choice(np.flatnonzero(values == values.max()))
            no, r, terminal = world.step(a)
            ret.append(r)
            o = no
        dret = discounted_return(ret, config.gamma)
        returns[ep] = dret
    return returns

def eval_opt(world, c51, counts, trial, config):
    # Policy Evaluation with c51, CVaR optimization
    returns = np.zeros(trial)
    for ep in range(trial):
        terminal = False
        o = world.reset()
        ret = []
        while not terminal:
            values = c51.CVaRopt(o, counts, c=config.args.opt,\
                    alpha=config.args.alpha, N=config.CVaRSamples)
            a = np.random.choice(np.flatnonzero(values == values.max()))
            no, r, terminal = world.step(a)
            ret.append(r)
            o = no
        dret = discounted_return(ret, config.gamma)
        returns[ep] = dret
    return returns

def discounted_return(ret, gamma):
    dret = 0
    for r in reversed(ret):
        dret = r + gamma * dret
    return dret
