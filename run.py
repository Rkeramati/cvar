import numpy as np

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

    if config.save_p: #Saving P matrix along the learning
        save_p_ep = int(args.num_episode/ config.save_p_total)

    # Make C51 Agent for evaluation and learning
    c51 = drl.C51(config, init='random',ifCVaR=True)
    if config.eval:
        c51_eval = drl.C51(config, init='random', ifCVaR=True)
    # init counts
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

            np.save(args.name + '_results_online_%d.npy'%(version), returns_online)
            if config.eval:
                np.save(args.name + '_results_eval_%d.npy'%(version), returns_eval)
            np.save(args.name + '_c51_p_%d.npy'%(version), c51.p)
            np.save(args.name + '_c51_counts_%d.npy'%(version), counts)
            if config.eval:
                np.save(args.name + '_c51_eval_p_%d.npy'%(version), c51_eval.p)
        # Save p matrix along the way
        if config.save_p:
            if ep%save_p_ep == 0:
                print('Saving P Mateirx, episode = %d'%(ep))
                np.save(args.name + '_c51_p_%d_%d.npy'%(version, ep//save_p_ep), c51.p)
                if config.eval:
                    np.save(args.name + '_c51_eval_p_%d_%d.npy'%(version, ep//save_p_ep), c51_eval.p)


if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(int(args.trial)):
        print('Trial: %d out of %d'%(i, args.trial))
        main(args=args, version=i)
