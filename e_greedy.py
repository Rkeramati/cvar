import numpy as np
import pickle

from core import drl
from config import *
from env import mrp # machine repair
from env import terrain as tr
from utils import utils

import argparse

parser = argparse.ArgumentParser(description='Argumens Parser')
parser.add_argument("--name", default= 'results/e_greedy', help='directory and prefix name for saving')
parser.add_argument("--trial", default=5, type=int, help='number of trials')
parser.add_argument("--alpha", default=0.25, type=float, help='CVaR risk value')
parser.add_argument("--env", default='mrp', help='envinronment')
parser.add_argument("--num_episode", type=int, default=100, help='number of episodes')
parser.add_argument("--egreedy", type=bool, default=True)
parser.add_argument("--option", type=int, default=1, help='e greedy scheduele')
parser.add_argument("--gamma", type=float, default=0.99, help='gamma')

def main(args, version):
    # Envinronments
    if args.env == 'mrp':
        world = mrp.machine_repair()
    elif args.env == '2D':
        world = tr.Nav2D(random=False)
    else:
        raise Exception("Envinronment not understood")

    # Config file
    config = Config(world.nS, world.nA)
    config.set(args)

    # Make C51 Agent for evaluation and learning
    c51 = drl.C51(config, init='random',ifCVaR=True)
    if config.e_greedy_eval:
        c51_eval = drl.C51(config, init='random', ifCVaR=True)
    # init counts
    num_evaluations = int(config.args.num_episode/ (config.eval_episode * 1.0))
    returns_online = np.zeros((num_evaluations, config.eval_trial))
    if config.e_greedy_eval:
        returns_eval = np.zeros((num_evaluations, config.eval_trial))

    for ep in range(config.args.num_episode):
        terminal = False
        lr = config.get_lr(ep)
        epsilon = config.get_epsilon(ep)

        # init world
        o = world.reset()

        # simulate
        while not terminal:
            if np.random.rand() <= epsilon:
                a = np.random.randint(world.nA)
            else:
                values = c51.CVaR(o, alpha=args.alpha, N=config.CVaRSamples)
                #a = np.argmax(values)
                a = np.random.choice(np.flatnonzero(values == values.max()))
            no, r, terminal = world.step(a)

            # update
            c51.observe(o, a, r, no, terminal, lr=lr, bonus=0.0)
            # Eval
            if config.e_greedy_eval:
                c51_eval.observe(o, a, r, no, terminal, lr=lr, bonus=0.0)

            # Go to next observation! I always forget this!
            o = no
        # Evaluate online
        # To evaluate the CVaR, we need the return distribution
        if ep%config.eval_episode == 0:
            eval_num = ep // config.eval_episode
            returns_online[eval_num, :] = utils.eval(world, c51, config.eval_trial, config)
            if config.e_greedy_eval:
                returns_eval[eval_num, :] = utils.eval(world, c51_eval, config.eval_trial, config)

        # Save:
        if ep%config.save_episode == 0:
            print('Saving results for episode %d out of %d, version %d'\
                    %(ep, config.args.num_episode, version))
            saveFile = {'p': c51.p, 'results': returns_online, 'gamma':config.gamma}
            pickle.dump(saveFile, open(args.name + '_version_%d_episode_%d.p', 'wb'))
            if config.e_greedy_eval:
                saveFile = {'p': c51_eval.p, 'results': returns_eval, 'gamma':config.gamma}
                pickle.dump(saveFile, open(args.name + '_eval_version_%d_episode_%5.p'%(version, ep),\
                        'wb'))

if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(int(args.trial)):
        print('Trial: %d out of %d'%(i, args.trial))
        main(args=args, version=i)

