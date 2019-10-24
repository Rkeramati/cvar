import tensorflow as tf
import gym
import numpy as np
import os
import sys
import pickle
import argparse

from iqnagent import IQNAgent
from iqnconfig import Config
from collections import deque
from hiv_env import HIVTreatment


parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("--alpha", type=float, default= 1, help='risk level')
parser.add_argument("--path", type=str, default="./IQN_HIV", help='save and load path')
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--load", type=bool, default=False, help="load model")
parser.add_argument("--num_episode", type=int, default=10000, help="num episodes")
parser.add_argument("--train_start", type=int, default=10, help="episode to start training")
# e-greedy
parser.add_argument("--option", type=int, default=5, help="epsilon schedule options")
# HIV:
parser.add_argument("--actionable_time_steps", type=int, default=5, help="number of actionable steps in the episode")
parser.add_argument("--normalize_state", type=bool, default=True, help="If normalize the state space to 0 and 1")
parser.add_argument("--normalize_reward", type=bool, default=True, help="If normalize the reward")
# stochasticity pattern
parser.add_argument("--st", type=int, default=5, help="stochasticity pattern")
parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")

def main(args):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    #env = gym.make('CartPole-v1')
    env = HIVTreatment(args)

    config = Config(env.state_space_dims, env.nA)
    config.set(args)

    with tf.Session() as sess:
        IQNbrain = IQNAgent(sess, config)

        if args.load:
            IQNbrain.load_model(tf.train.latest_checkpoint(args.path+'/models'))
        else:
            sess.run(tf.global_variables_initializer())
            returns = np.zeros((args.num_episode, 2))
            #evaluation_returns = np.zeros((int(args.num_episode/Config.eval_episode) + 1, args.eval))
            #all_episode_reward = np.zeros((Config.args.num_episode, env.episodeCap))


        for ep in range(args.num_episode):
            terminal = False

            episode_return = []; episode_loss = []
            s = env.reset()
            epsilon = config.get_epsilon(ep)
            learning_rate = config.get_lr(ep)

            count = 0
            while not terminal:
                count += 1
                action, actions_value, q_dist, tau_beta = IQNbrain.choose_action(s, epsilon)
                next_s, reward, terminal, _ = env.step(action)
                IQNbrain.memory_add(s, float(action), reward, next_s, int(terminal))
                s = next_s
                episode_return.append(reward)
                # Training
                if ep > args.train_start:
                    loss = IQNbrain.learn(learning_rate)
                    episode_loss.append(loss)
            
            returns[ep, 0] = discounted_return(episode_return, config.gamma)
            returns[ep, 1] = np.mean(episode_loss)
            if ep%config.print_episode == 0:
                print("Episode:{} | Episode Reward:{} | Episode Loss:{}".\
                        format(ep, returns[ep, 0], returns[ep, 1]))
                        
            if ep%config.save_episode == 0:
                save_file = {'ep': ep, 'returns': returns, 'episode_data': env.episode_data}
                pickle_in = open(args.path + '/results/episode_%d.p'%(ep), 'wb')
                pickle.dump(save_file, pickle_in)
                pickle_in.close()
                # Neural Net Saving
                ckpt_path = os.path.join(args.path + '/models', 'IQN.ckpt')
                IQNbrain.save_model(ckpt_path)

def discounted_return(returns, gamma):
    ret = 0
    for r in reversed(returns):
        ret = r + gamma * ret
    return ret


if __name__ == "__main__":
    args = parser.parse_args()
    args.num_episode += 1
    #args.save_name = '/next/u/keramati' + args.path
    if not os.path.exists(args.path + "/results"):
        os.makedirs(args.path + "/results")
    if not os.path.exists(args.path + "/models"):
        os.makedirs(args.path + "/models")
    main(args)