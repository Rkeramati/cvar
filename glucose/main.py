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
from core import config, drl, replay, prob

# Others
#import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from datetime import datetime
import pickle
import gym
from gym.envs.registration import register
import tensorflow as tf
import os

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

def run_egreedy(args):
    print('[*] Run E greedy')
    # Running egreedy
    # Set optimism to zero:
    args.opt = 0
    with tf.Session() as sess:
        env = make_env(args)
        Config = config.config(env, args)

        if args.load_name is not None:
            load_file = pickle.load(open(args.load_name + '.p', 'rb'))
            replay_buffer = replay.Replay(Config, load=True, name=args.load_name)
            C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, memory=replay_buffer)
            returns = load_file["returns"]
            initial_ep = load_file["ep"]
            saver = tf.train.Saver()
            saver.restore(sess, args.load_name + '.ckpt')
            print("[*] TF model restored")
        else:
            returns = np.zeros((Config.args.num_episode, 2))
            replay_buffer = replay.Replay(Config, load=False)
            C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, memory=replay_buffer)
            initial_ep = 0
            saver = tf.train.Saver()
            sess.run(tf.initializers.global_variables())
            print("[*] TF model initialized")

        # Set up the summary writer
        summary_writer = tf.summary.FileWriter(args.save_name + '/summary', sess.graph)

        # Training Loop:
        C51_loss = []
        train_step = 0
        for ep in range(initial_ep, Config.args.num_episode+initial_ep):
            terminal = False
            step = 0
            # TODO: Pass the lr to Adam
            lr = Config.get_lr(ep)

            # Epsilon for e-greedy:
            if ep%Config.eval_episode == 0:
                epsilon=0 # Evaluation Episode, no epsilon
            else:
                epsilon = Config.get_epsilon(ep)

            episode_return = []
            Config.max_step = args.hour*60/(env.env.sensor.sample_time) # Compute the max step
            meal = 0
            observation = Config.process(env.reset(), meal=meal) # Process will add stochasticity
                                                                 # to the observed state
            while step <= Config.max_step and not terminal:
                if np.random.rand() <= epsilon and args.e_greedy:
                    action_id = np.random.randint(Config.nA)
                else:
                    if Config.args.ifCVaR:
                        o = np.expand_dims(observation, axis=0)
                        counts = np.ones((1, Config.nA))
                        distribution = C51.predict(sess, o)
                        # args.opt = 0
                        values = C51.CVaRopt(distribution, count=counts,\
                                alpha=Config.args.alpha, N=Config.CVaRSamples, c=args.opt, bonus=0.0)
                    else:
                        raise Exception("Not Implemented!")
                        values = C51.Q(observation)
                    action_id = np.random.choice(np.flatnonzero(values == values.max()))
                action = Config.get_action(action_id) # get action with/ without randomness
                delay = Config.get_delay()
                next_observation, reward, terminal, info, num_step = custom_step(env,\
                        action, step, Config.max_step, delay)

                step += num_step
                BG = next_observation.CGM
                next_observation = Config.process(next_observation, meal=info['meal'])
                no = np.expand_dims(next_observation, axis=0)
                next_counts = counts # hack to avoind passing counts
                episode_return.append(reward)
                if step >= Config.max_step:
                    terminal = True
                # TODO: egreedy eval ep should not be trained on
                replay_buffer.add(observation, action_id, reward, terminal,\
                        counts, next_counts)
                # Training:
                l, summary = C51.train(sess=sess, size=Config.train_size, opt=args.opt)

                if ep%Config.summary_write_episode == 0 and summary is not None:
                    summary_writer.add_summary(summary, train_step)
                train_step += 1
                if l is not None:
                    C51_loss.append(l)
                    returns[ep, 1] = l
                observation = next_observation

            returns[ep, 0] = discounted_return(episode_return, Config.args.gamma)
            if ep%Config.print_episode == 0 and not ep%Config.eval_episode==0:
                print("Training.  Episode ep:%3d, Discounted Return = %g, Epsilon = %g, BG=%g, C51 average loss=%g"\
                        %(ep, returns[ep, 0], epsilon, BG, np.mean(C51_loss)))
            if ep % Config.eval_episode == 0:
                print("Evaluation Episode ep:%3d, Discounted Return = %g, BG = %g"\
                        %(ep, returns[ep, 0], BG))
            if ep% Config.save_episode == 0:
                save_file = {'ep': ep, 'returns': returns}
                replay_buffer.save(args.save_name)
                pickle_in = open(args.save_name + '.p', 'wb')
                pickle.dump(save_file, pickle_in)
                pickle_in.close()
                saver.save(sess, args.save_name + '.ckpt')


def run(args):
    with tf.Session() as sess:
        env = make_env(args)
        Config = config.config(env, args)

        if args.load_name is not None:
            load_file = pickle.load(open(args.load_name + '.p', 'rb'))
            replay_buffer = replay.Replay(Config, load=True, name=args.load_name)
            C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, memory=replay_buffer)
            Counts = prob.LogProb(Config)
            returns = load_file["returns"]
            initial_ep = load_file["ep"]
            saver = tf.train.Saver()
            saver.restore(sess, args.load_name + '.ckpt')
            print("[*] TF model restored")
        else:
            returns = np.zeros((Config.args.num_episode, 2))
            replay_buffer = replay.Replay(Config, load=False)
            C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, memory=replay_buffer)
            Counts = prob.LogProb(Config)
            initial_ep = 0
            saver = tf.train.Saver()
            sess.run(tf.initializers.global_variables())
            print("[*] TF model initialized")

        summary_writer = tf.summary.FileWriter(args.save_name + '/summary', sess.graph)

        # Training Loop:
        C51_loss = []
        train_step = 0
        for ep in range(initial_ep, Config.args.num_episode+initial_ep):
            terminal = False
            step = 0
            # TODO: Currently using fixed lr for Adam
            lr = Config.get_lr(ep)

            episode_return = []
            Config.max_step = args.hour*60/(env.env.sensor.sample_time) # Compute the max step
            meal = 0
            observation = Config.process(env.reset(), meal=meal) # Process will add stochasticity
                                                                 # to the observed state
            while step <= Config.max_step and not terminal:
                if Config.args.ifCVaR:
                    o = np.expand_dims(observation, axis=0)
                    counts, _ = Counts.compute_counts(sess, o)
                    counts = np.array(counts)
                    distribution = C51.predict(sess, o)
                    c = np.expand_dims(counts, axis=0)
                    values = C51.CVaRopt(distribution, count=c,\
                                alpha=Config.args.alpha, N=Config.CVaRSamples, c=args.opt, bonus=0.0)
                else:
                    raise Exception("Not Implemented!")
                    values = C51.Q(observation)
                action_id = np.random.choice(np.flatnonzero(values == values.max()))
                action = Config.get_action(action_id) # get action with/ without randomness
                delay = Config.get_delay()
                next_observation, reward, terminal, info, num_step = custom_step(env,\
                        action, step, Config.max_step, delay)

                step += num_step
                BG = next_observation.CGM
                next_observation = Config.process(next_observation, meal=info['meal'])
                no = np.expand_dims(next_observation, axis=0)
                next_counts, counts_summary = Counts.compute_counts(sess, no)
                next_counts = np.array(next_counts)
                episode_return.append(reward)
                if step >= Config.max_step:
                    terminal = True

                replay_buffer.add(observation, action_id, reward, terminal,\
                        counts, next_counts)
                # Training:
                l, summary = C51.train(sess=sess, size=Config.train_size, opt=args.opt)
                summary_counts = Counts.train(sess, o, np.expand_dims(action, axis=0))

                if ep%Config.summary_write_episode == 0 and summary is not None:
                    summary_writer.add_summary(summary, train_step)
                    summary_writer.add_summary(summary_counts, train_step)
                    summary_writer.add_summary(counts_summary, train_step)
                train_step += 1
                if l is not None:
                    C51_loss.append(l)
                    returns[ep, 1] = l
                observation = next_observation


            returns[ep, 0] = discounted_return(episode_return, Config.args.gamma)
            if ep%Config.print_episode == 0 and not ep%Config.eval_episode==0:
                print("Training.  Episode ep:%3d, Discounted Return = %g, Epsilon = %g, BG=%g, C51 average loss=%g"\
                        %(ep, returns[ep, 0], epsilon, BG, np.mean(C51_loss)))
            if ep % Config.eval_episode == 0:
                print("Evaluation Episode ep:%3d, Discounted Return = %g, BG = %g"\
                        %(ep, returns[ep, 0], BG))
            if ep% Config.save_episode == 0:
                save_file = {'ep': ep, 'returns': returns}
                replay_buffer.save(args.save_name)
                pickle_in = open(args.save_name + '.p', 'wb')
                pickle.dump(save_file, pickle_in)
                pickle_in.close()
                saver.save(sess, args.save_name + '.ckpt')

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.save_name + '/summary'):
            os.makedirs(args.save_name + '/summary')
    if args.e_greedy:
        args.opt = 0
        run_egreedy(args)
    else:
        run(args)

