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
# import matplotlib.pyplot as plt
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
    env = gym.make('simglucose-adult3-v0')
    return env

def _step(env, action, step, max_step):
    reward = []
    num_step = 1
    obs, rew, terminal, info = env.step(action)
    reward.append(rew)
    meal = info['meal']

    while meal <= 0 and not terminal and step + num_step <= max_step:
        obs, rew, terminal, info = env.step([0, 0])
        meal = info['meal']
        num_step += 1
        reward.append(rew)
    return obs, np.mean(reward), terminal, info, num_step

def run(args):
    env = make_env(args)
    Config = config.config(env, args)

    if args.load_name is not None:
        load_file = pickle.load(open(args.load_name + '.p'), 'rb')
        replay_buffer = replay.Replay(Config, load=True, name=args.load_name)
        C51 = drl.C51(Config, ifCVaR=False, p=load_file["p"], memory=replay_buffer)
        returns = load_file["returns"]
        initial_ep = load_file["ep"]
    else:
        returns = np.zeros(Config.args.num_episode)
        replay_buffer = replay.Replay(Config, load=False)
        C51 = drl.C51(Config, ifCVaR=False, p=None, memory=replay_buffer)
        initial_ep = 0

    # Training Loop:
    for ep in range(initial_ep, Config.args.num_episode+initial_ep):
        terminal = False
        step = 0
        lr = Config.get_lr(ep)

        if ep%Config.eval_episode == 0:
            epsilon=0
        else:
            epsilon = Config.get_epsilon(ep)

        episode_return = []
        Config.max_step = args.hour*60/(env.env.sensor.sample_time) # Compute the max step
        meal = 0
        observation = Config.process(env.reset(), meal=0)

        while step <= Config.max_step and not terminal:

            if np.random.rand() <= epsilon:
                action_id = np.random.randint(Config.nA)
            else:
                values = C51.Q(observation)
                action_id = np.random.choice(np.flatnonzero(values == values.max()))
            action = get_action(action_id, Config.action_map)
            next_observation, reward, terminal, info, num_step = _step(env,\
                    action, step, Config.max_step)
            step += num_step
            BG = next_observation.CGM
            next_observation = Config.process(next_observation, meal=info['meal'])

            episode_return.append(reward)
            if step >= Config.max_step:
                terminal = True
            #print("add observation:", observation, action_id, reward, next_observation, terminal, step)
            replay_buffer.add(observation, action_id, reward, terminal)
            C51.train(size=Config.train_size, lr=lr, counts=None, opt=0.0)

            observation = next_observation

            #if args.animation:
            #    env.render()
            #    plt.pause(0.001)
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

