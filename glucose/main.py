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
parser.add_argument("--load_name", type=str, default= None, help='directory and prefix name for loading')
parser.add_argument("--save_name", type=str, default= 'results/c51', help='directory and prefix name for saving')
parser.add_argument("--animation", type=bool, default=False, help="To show the animation")
parser.add_argument("--hour", type=int, default=2, help="Simulation hour")
parser.add_argument("--seed", type=int, default=2, help="random seed number")

def make_env(args):
    register(
    id='simglucose-adolescent2-v0',
    entry_point='utils.gymenv:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002',
            'reward_fun': reward_fun,
            'done_fun': done_fun,
            'scenario': scenario_fun(),
            'seed':args.seed} # Returning a custom scenario
    )
    env = gym.make('simglucose-adolescent2-v0')
    return env

def run(args, save_name,load_name):
    env = make_env(args)
    Config = config.config(env)
    if load_name is not None:
        load_file = pickle.load(open(load_name + '.p'), 'rb')
        C51 = drl.C51(Config, ifCVaR=False, p=load_file["p"])
        returns = load_file["returns"]
        initial_ep = load_file["ep"]
    else:
        C51 = drl.C51(Config, ifCVaR=False, p=None)
        returns = np.zeros(Config.num_episode)
        initial_ep = 0

    # Training Loop:
    for ep in range(initial_ep, Config.num_episode+initial_ep):
        terminal = False
        step = 0
        lr = Config.get_lr(ep)

        if ep%Config.eval_episode == 0:
            epsilon=0
        else:
            epsilon = Config.get_epsilon(ep)

        observation = Config.process(env.reset(), meal=0)
        episode_return = []
        Config.max_step = args.hour*60/(env.env.sensor.sample_time) # Compute the max step
        while step <= Config.max_step and not terminal:
            # Pick action
            if np.random.rand() <= epsilon:
                action_id = np.random.randint(Config.nA)
            else:
                values = C51.Q(observation)
                action_id = np.random.choice(np.flatnonzero(values == values.max()))
            action = get_action(action_id, Config.action_map)

            next_observation, reward, terminal, info = env.step(action)

            next_observation = Config.process(next_observation, meal=info['meal'])

            if step == Config.max_step:
                terminal = True
            C51.observe(observation, action_id, reward, next_observation, terminal, lr=lr, bonus=0.0)

            observation = next_observation
            step += 1
            episode_return.append(reward)
            if args.animation:
                env.render()
                plt.pause(0.001)
        returns[ep] = discounted_return(episode_return, Config.gamma)
        if ep%Config.print_episode == 0 and not ep%Config.eval_episode==0:
            print("Training.  Episode ep:%3d, Discounted Return = %g, Epsilon = %g"%(ep, returns[ep], epsilon))
        if ep % Config.eval_episode == 0:
            print("Evaluation Episode ep:%3d, Discounted Return = %g"%(ep, returns[ep]))
        if ep% Config.save_episode == 0:
            save_file = {'p': C51.p, 'ep': ep, 'returns': returns}
            pickle.dump(save_file, open(args.save_name + '.p', 'wb'))

if __name__ == "__main__":
    args = parser.parse_args()
    run(args, args.save_name, args.load_name)

