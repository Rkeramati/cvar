import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil
from timeit import default_timer as timer
from datetime import timedelta

from iqnagent import IQNAgent
from iqnconfig import Config
from collections import deque
import argparse


parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("--eta", type=float, default= 0.25, help='risk level')

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

env = gym.make('CartPole-v1')
env.seed(seed)

config = Config(env.observation_space.shape[0], env.action_space.n)

MINIBATCH_SIZE = 32
TRAIN_START = 1000
TARGET_UPDATE = 25
MEMORY_SIZE = 20000
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
LEARNING_RATE = 0.0001
DISCOUNT = 0.99
LOAD_model = False
SAVE_model = True
TRAIN = True
RENDER = False
VIEW_DIST = False

path = "./CartPole_iqn_test"


def plot_cdf(actions_value, q_dist, tau_beta):
    y = np.repeat(np.sort(tau_beta), 2, 0)
    plt.ylim([0, 1])
    plt.xlim([np.max(actions_value)-1, np.max(actions_value)+1])
    plt.xlabel('Q')
    plt.ylabel('CDF')
    plt.step(np.transpose(np.sort(q_dist[0])), np.transpose(y))
    plt.legend(labels=('Left', 'Right'))
    plt.draw()
    plt.pause(0.00001)
    plt.clf()


def train(args):
    config.eta = args.eta
    with tf.Session() as sess:
        IQNbrain = IQNAgent(sess, config)

        if LOAD_model:
            IQNbrain.load_model(tf.train.latest_checkpoint(path))
        else:
            sess.run(tf.global_variables_initializer())

        all_rewards = []
        frame_rewards = []
        loss_list = []
        loss_frame = []
        recent_rlist = deque(maxlen=15)
        recent_rlist.append(0)
        episode, epoch, frame = 0, 0, 0
        start = timer()

        while np.mean(recent_rlist) < 499:
            episode += 1

            rall, count = 0, 0
            done = False
            s = env.reset()

            while not done:
                if RENDER:
                    env.render()

                frame += 1
                count += 1

                action, actions_value, q_dist, tau_beta = IQNbrain.choose_action(s)

                if VIEW_DIST:
                    plot_cdf(actions_value, q_dist, tau_beta)

                s_, r, done, l = env.step(action)

                if done and count >= 500:
                    reward = 1
                elif done and count < 500:
                    reward = -10
                else:
                    reward = 0

                IQNbrain.memory_add(s, float(action), reward, s_, int(done))
                s = s_

                rall += r

                if frame > TRAIN_START and TRAIN:
                    loss = IQNbrain.learn()
                    loss_list.append(loss)
                    loss_frame.append(frame)

            recent_rlist.append(rall)
            all_rewards.append(rall)
            frame_rewards.append(frame)

            print("Episode:{} | Frames:{} | Reward:{} | Recent reward:{}".\
                format(episode, frame, rall, np.mean(recent_rlist)))
            print(np.mean(loss_list))
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
        ckpt_path = os.path.join(path, 'IQN.ckpt')
        if SAVE_model:
            IQNbrain.save_model(ckpt_path)

        plt.figure(figsize=(10, 8))
        plt.subplot(211)
        plt.title('Episode %s. Recent_reward: %s. Time: %s' % (
            len(all_rewards), np.mean(recent_rlist), timedelta(seconds=int(timer() - start))))
        plt.plot(frame_rewards, all_rewards)
        plt.ylim(0, 510)
        plt.subplot(212)
        plt.title('Loss')
        plt.plot(loss_frame, loss_list)
        #plt.ylim(0, 20)
        plt.show()
        plt.close()


def test():
    with tf.Session() as sess:
        IQNbrain = IQNAgent(sess, config)

        IQNbrain.load_model(tf.train.latest_checkpoint(path))

        masspole_list = np.arange(0.01, 0.21, 0.025)
        length_list = np.arange(0.5, 3, 0.25)

        performance_mtx = np.zeros([masspole_list.shape[0], length_list.shape[0]])

        for im in range(masspole_list.shape[0]):
            for il in range(length_list.shape[0]):
                env.env.masspole = masspole_list[im]
                env.env.length = length_list[il]

                all_rewards = []

                for episode in range(5):

                    rall, count = 0, 0
                    done = False
                    s = env.reset()

                    while not done:
                        if RENDER:
                            env.render()

                        action, actions_value, q_dist, tau_beta = IQNbrain.choose_action(s)

                        s_, r, done, _ = env.step(action)

                        s = s_

                        rall += r

                    all_rewards.append(rall)

                    print("Episode:{} | Reward:{} ".format(episode, rall))

                performance_mtx[im, il] = np.mean(all_rewards)

        fig, ax = plt.subplots()
        ims = ax.imshow(performance_mtx, cmap=cm.gray, interpolation=None, vmin=0, vmax=500)
        ax.set_xticks(np.arange(0, length_list.shape[0], length_list.shape[0] - 1))
        ax.set_xticklabels(['0.5', '3'])
        ax.set_yticks(np.arange(0, masspole_list.shape[0], masspole_list.shape[0] - 1))
        ax.set_yticklabels(['0.01', '0.20'])
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Pole mass')
        ax.set_title('Robustness test: IQN')
        fig.colorbar(ims, ax=ax)

        plt.show()
        plt.close()


def main(args):
    if TRAIN:
        train(args)
    else:
        test()


if __name__ == "__main__":
    args = parser.parse_args()
    # args.num_episode += 1
    # print('[*] If on server uncomment the saving part')
    # #args.save_name = '/next/u/keramati' + args.save_name
    # if not os.path.exists(args.save_name + "/summary"):
    #     os.makedirs(args.save_name + "/summary")
    # # Run the egreedy
    main(args)