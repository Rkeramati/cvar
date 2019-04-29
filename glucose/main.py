# Simglucose Imports
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from simglucose.controller.base import Controller, Action

# Custom imports
from utils.custom import *
from utils.gymenv import T1DSimEnv
from core import config, drl, replay

# Others
#import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pickle
import gym
from gym.envs.registration import register

import argparse

parser = argparse.ArgumentParser(description='Argumens Parser')
parser.add_argument("--load_name", type=str, default= None, help='directory and prefix name for loading')
parser.add_argument("--save_name", type=str, default= 'results/c51', help='directory and prefix name for saving')
parser.add_argument("--animation", type=bool, default=False, help="To show the animation")
parser.add_argument("--hour", type=int, default=2, help="Simulation hour")
parser.add_argument("--seed", type=int, default=2, help="random seed number")
parser.add_argument("--gamma", type=float, default=0.999, help="Discount Factor")
parser.add_argument("--num_episode", type=int, default=1000, help="Number of episodes")
parser.add_argument("--delta_state", type=float, default=0.0, help="stochasticity in the state")
parser.add_argument("--action_sigma",type=float, default=0.0, help="action stochasticity")
parser.add_argument("--ifCVaR", type=bool, default=False, help="if optimize for CVaR")
parser.add_argument("--alpha", type=float, default=0.25, help="CVaR risk value")
parser.add_argument("--action_delay", type=int, default=0, help="maximum number of steps to delay the action")
parser.add_argument("--e_greedy", type=bool, default=False, help="if e-greedy")
parser.add_argument("--opt", type=float, default=1.0, help="optimism")

def make_env(args):
    register(
    id='simglucose-adult3-v0',
    entry_point='utils.gymenv:T1DSimEnv',
    kwargs={'patient_name': 'adult#003',
            'reward_fun': reward_fun,
            'done_fun': done_fun,
            'scenario': scenario_fun(),
            'seed':args.seed} # Returning a custom scenario
    )
    # Check if delay make sense:
    env = gym.make('simglucose-adult3-v0')
    if minDiff(env.env.sensor.sample_time) <= args.action_delay:
        raise Exception("Too much delay in action, forget a whole meal!")

    return env

def _step(env, action, step, max_step, delay):
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

def run(args):
    env = make_env(args)
    Config = config.config(env, args)
    counts = np.zeros((Config.nS, Config.nA)) + 1

    if args.load_name is not None:
        load_file = pickle.load(open(args.load_name + '.p'), 'rb')
        replay_buffer = replay.Replay(Config, load=True, name=args.load_name)
        C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, p=load_file["p"], memory=replay_buffer)
        returns = load_file["returns"]
        initial_ep = load_file["ep"]
    else:
        returns = np.zeros(Config.args.num_episode)
        replay_buffer = replay.Replay(Config, load=False)
        C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, p=None, memory=replay_buffer)
        initial_ep = 0

    # Training Loop:
    for ep in range(initial_ep, Config.args.num_episode+initial_ep):
        terminal = False
        step = 0
        lr = Config.get_lr(ep)

        # Epsilon for e-greedy:
        if ep%Config.eval_episode == 0:
            epsilon=0
        else:
            epsilon = Config.get_epsilon(ep)

        episode_return = []
        Config.max_step = args.hour*60/(env.env.sensor.sample_time) # Compute the max step
        meal = 0
        observation = Config.process(env.reset(), meal=0) # Process will add stochasticity
                                                          # to the observed state
        while step <= Config.max_step and not terminal:

            if np.random.rand() <= epsilon and args.e_greedy:
                action_id = np.random.randint(Config.nA)
            else:
                if Config.args.ifCVaR:
                    o = np.expand_dims(observation, axis=-1)
                    values = C51.CVaRopt(o, count=counts,\
                            alpha=Config.args.alpha, N=Config.CVaRSamples, c=args.opt, bonus=0.0)
                else:
                    values = C51.Q(observation)
                action_id = np.random.choice(np.flatnonzero(values == values.max()))

            action = Config.get_action(action_id) # get action with/ without randomness

            delay = Config.get_delay()
            next_observation, reward, terminal, info, num_step = _step(env,\
                    action, step, Config.max_step, delay)

            counts[observation, action_id] += 1
            step += num_step
            BG = next_observation.CGM
            next_observation = Config.process(next_observation, meal=info['meal'])

            episode_return.append(reward)
            if step >= Config.max_step:
                terminal = True

            replay_buffer.add(observation, action_id, reward, terminal)
            C51.train(size=Config.train_size, lr=lr, counts=counts, opt=args.opt)

            observation = next_observation

        returns[ep] = discounted_return(episode_return, Config.args.gamma)
        if ep%Config.print_episode == 0 and not ep%Config.eval_episode==0:
            print("Training.  Episode ep:%3d, Discounted Return = %g, Epsilon = %g, BG=%g"\
                    %(ep, returns[ep], epsilon, BG))
        if ep % Config.eval_episode == 0:
            print("Evaluation Episode ep:%3d, Discounted Return = %g, BG = %g"%(ep, returns[ep], BG))
        if ep% Config.save_episode == 0:
            save_file = {'p': C51.p, 'ep': ep, 'returns': returns}
            pickle.dump(save_file, open(args.save_name + '.p', 'wb'))

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)

