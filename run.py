import numpy as np
import pickle

from core import drl
from config import *
from env import mrp # machine repair
from env import terrain as tr
from utils import utils

import argparse

parser = argparse.ArgumentParser(description='Argumens Parser')
parser.add_argument("--name", default= 'results/cvar_opt', help='directory and prefix name for saving')
parser.add_argument("--trial", default=5, type=int, help='number of trials')
parser.add_argument("--opt", default=1.0, type=float, help='optimism constant')
parser.add_argument("--alpha", default=0.25, type=float, help='CVaR risk value')
parser.add_argument("--env", default='mrp', help='envinronment')
parser.add_argument("--num_episode", type=int, default=100, help='number of episodes')
parser.add_argument("--egreedy", type=bool, default=False, help='If egreedy')
parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
parser.add_argument("--load", type=str, default=None, help="Loading Address")

def main(args, version):
    print("Warning Loading for Evaluation not implemented!!")

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
    if args.load is not None:
        load_file = pickle.load(open(args.load, 'rb'))
        # Make C51 Agent for evaluation and learning
        c51 = drl.C51(config, init='random',ifCVaR=True, p=load_file['p'])
    else:
        c51 = drl.C51(config, init='random', ifCVaR = True, p=None)

    if config.eval:
        c51_eval = drl.C51(config, init='random', ifCVaR=True)
    # init counts
    if args.load is not None:
        counts = load_file['counts']
        print("Loading Counts")
    else:
        counts = np.zeros((world.nS, world.nA)) + 1 # 1 for all state-action pair

    num_evaluations = int(config.args.num_episode/ (config.eval_episode * 1.0))
    returns_online = np.zeros((num_evaluations, config.eval_trial))
    if config.eval:
        returns_eval = np.zeros((num_evaluations, config.eval_trial))

    for ep in range(config.args.num_episode):
        terminal = False
        lr = config.get_lr(ep)

        # init world
        o = world.reset()

        # simulate
        while not terminal:
            values = c51.CVaRopt(o, counts, c=args.opt, alpha=args.alpha, N=config.CVaRSamples)
            a = np.random.choice(np.flatnonzero(values == values.max()))
            no, r, terminal = world.step(a)
            counts[o, a] += 1

            # update
            c51.observe(o, a, r, no, terminal, lr=lr, bonus=args.opt/np.sqrt(counts[o,a]))
            # Eval
            if config.eval:
                c51_eval.observe(o, a, r, no, terminal, lr, bonus=0.0)

            # Go to next observation! I always forget this!
            o = no
        # Evaluate online
        # To evaluate the CVaR, we need the return distribution
        if ep%config.eval_episode == 0:
            eval_num = ep // config.eval_episode
            # For online evaluation, counts and optimism bonus should be passed
            returns_online[eval_num, :] = utils.eval_opt(world, c51, counts, config.eval_trial, config)
            if config.eval:
                returns_eval[eval_num, :] = utils.eval(world, c51_eval, config.eval_trial, config.gamma)

        # Save:
        if ep%config.save_episode == 0:
            print('Saving results for episode %d out of %d, version %d'\
                    %(ep, config.args.num_episode, version))

            saveFile = {'p': c51.p, 'counts': counts, 'episode': ep, 'gamma': config.gamma,\
                    'returns': returns_online}
            pickle.dump(saveFile, open(args.name + "_trail_%d_episode_%d.p"%(version, ep), "wb"))
            if config.eval:
                saveFile = {'p': c51_eval.p, 'counts': counts, 'episode': ep, 'gamma': config.gamma,\
                        'returns': returns_eval}
                pickle.dump(saveFile, open(args.name + "_eval__trial_%d_episode_%d.p"%(version, ep)\
                        , "wb"))

if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(int(args.trial)):
        print('Trial: %d out of %d'%(i, args.trial))
        main(args=args, version=i)
