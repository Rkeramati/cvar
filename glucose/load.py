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
from core import config, drl

# Others
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pickle
import gym
from gym.envs.registration import register

import argparse
parser = argparse.ArgumentParser(description='Argumens Parser')
parser.add_argument("--load", type=str, default=None, help='load file address')
parser.add_argument("--save", type=str, default=None, help='save file address')
parser.add_argument("--hour", type=int, default=2, help="Simulation hour")
parser.add_argument("--seed", type=int, default=2, help="random seed number")
parser.add_argument("--gamma", type=float, default=0.999, help="Discount Factor")
parser.add_argument("--num_simulations", type=int, default=100, help="Number of simulations")
parser.add_argument("--delta_state", type=float, default=0.0, help="stochasticity in the state")
parser.add_argument("--action_sigma",type=float, default=0.0, help="action stochasticity")
parser.add_argument("--ifCVaR", type=bool, default=False, help="if optimize for CVaR")
parser.add_argument("--alpha", type=float, default=0.25, help="CVaR risk value")
parser.add_argument("--action_delay", type=int, default=0, help="maximum number of steps to delay the   action")


def make_env(args):
    register(
    id='simglucose-adult3-v0',
    entry_point='utils.gymenv:T1DSimEnv',
    kwargs={'patient_name': 'adult#003',
            'reward_fun': reward_fun,
            'done_fun': done_fun,
            'scenario': scenario_fun(),
           'seed': args.seed} # Returning a custom scenario
    )
    env = gym.make('simglucose-adult3-v0')
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
    load_file = pickle.load(open(args.load, 'rb'))
    env = make_env(args)
    Config = config.config(env, args)
    C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, p=load_file["p"])

    returns = []
    for test in range(args.num_simulations):
        terminal = False
        step = 0
        episode_return = []
        Config.max_step = args.hour*60/(env.env.sensor.sample_time) # Compute the max step
        meal = 0
        observation = Config.process(env.reset(), meal=0)
        o = np.expand_dims([observation], axis=1)
        counter = 0
        while step <= Config.max_step and not terminal:
            if args.ifCVaR:
                values = C51.CVaRopt(o, count=None,\
                             alpha=Config.args.alpha, N=Config.CVaRSamples, c=0.0, bonus=0.0)
            else: #Expectation
                values = C51.Q(observation)
            action_id = np.random.choice(np.flatnonzero(values == values.max()))
            action = Config.get_action(action_id)

            delay = Config.get_delay()

            next_observation, reward, terminal, info, num_step = _step(env,\
                action, step, Config.max_step, delay)
            step += num_step

            next_observation = Config.process(next_observation, meal=info['meal'])

            episode_return.append(reward)
            if step >= Config.max_step:
                terminal = True
            observation = next_observation
            o = np.expand_dims([observation], axis=1)
            counter += 1
        returns.append(discounted_return(episode_return, 0.999))
        print('test: %d, return: %0.5g'%(test, returns[-1]))
    np.save(args.save+'.npy', returns)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)

