import numpy as np
import tensorflow as tf
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("--load_name", type=str, default= None, help='directory and prefix name for      loading')
parser.add_argument("--load_ep", type=int, default=0, help='load episode')
parser.add_argument("--save_name", type=str, default= 'results/egreedy/c51', help='directory and prefix name for saving')
parser.add_argument("--seed", type=int, default=1, help="random seed number")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount Factor")
parser.add_argument("--num_episode", type=int, default=5000, help="Number of episodes")
parser.add_argument("--ifCVaR", type=bool, default=False, help="if optimize for CVaR")
parser.add_argument("--alpha", type=float, default=0.25, help="CVaR risk value")
# envinronment specific arguments
parser.add_argument("--actionable_time_steps", type=int, default=20, help="number of actionable steps in the episode")
parser.add_argument("--normalize_state", type=bool, default=True, help="If normalize the state space to 0 and 1")
parser.add_argument("--normalize_reward", type=bool, default=True, help="If normalize the reward")
# optimism amount, should be zero for egreedy
parser.add_argument("--opt", type=float, default=0.0, help="opt amount")
# Tuning Parameters
parser.add_argument("--arch", type=int, default=1, help="architecture type")
# stochasticity pattern
parser.add_argument("--st", type=int, default=1, help="stochasticity pattern")
parser.add_argument("--eval", default =20, help="number of runs for each evaluation")

from core import config, drl, replay, prob
from utils.hiv_env import *
import os

def egreedy(args):
    print('[*] Run e-greedy')

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    with tf.Session() as sess:
        env = HIVTreatment(args)
        Config = config.config(env, args)
        Config.max_step = env.episodeCap

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
            evaluation_returns = np.zeros((int(args.num_episode/Config.eval_episode) + 1, args.eval))
            replay_buffer = replay.Replay(Config, load=False)
            counts = np.ones((1, Config.nA)) #hack to avoid re-writing code
            C51 = drl.C51(Config, ifCVaR=Config.args.ifCVaR, memory=replay_buffer)
            saver = tf.train.Saver()
            sess.run(tf.initializers.global_variables())
            print("[*] TF model initialized")

        C51_loss = []
        train_step = 0
        number_of_evaluations = 0
        for ep in range(Config.args.num_episode):
            terminal = False
            lr = Config.get_lr(ep) # lr is adaptive in this env, not similar to glucose
            epsilon = Config.get_epsilon(ep)

            # Evaluation Runs for evaluation
            if ep%Config.eval_episode == 0:
                for eval_episode in range(args.eval):
                    terminal = False
                    episode_return = []
                    observation = env.reset()

                    while not terminal:
                        if Config.args.ifCVaR:
                            o = np.expand_dims(observation, axis=0)
                            counts = np.ones((1, Config.nA))
                            distribution = C51.predict(sess, o)
                            values = C51.CVaRopt(distribution, count=counts,\
                                 alpha=Config.args.alpha, N=Config.CVaRSamples, c=args.opt, bonus=0.0)
                        else:
                            o = np.expand_dims(observation, axis=0)
                            distribution = C51.predict(sess, o)
                            values = C51.Q(distribution)
                        action_id = np.random.choice(np.flatnonzero(values == values.max()))
                        next_observation, reward, terminal, info = env.step(action_id)
                        no = np.expand_dims(next_observation, axis=0)
                        next_counts = counts # hack to avoind passing counts
                        episode_return.append(reward)
                        observation = next_observation
                    evaluation_returns[number_of_evaluations, eval_episode] =\
                            discounted_return(episode_return, Config.args.gamma)
                number_of_evaluations += 1
            episode_return = []
            observation = env.reset()

            while not terminal:
                if np.random.rand() <= epsilon:
                    action_id = np.random.randint(Config.nA)
                else:
                    if Config.args.ifCVaR:
                        o = np.expand_dims(observation, axis=0)
                        counts = np.ones((1, Config.nA))
                        distribution = C51.predict(sess, o)
                        values = C51.CVaRopt(distribution, count=counts,\
                                 alpha=Config.args.alpha, N=Config.CVaRSamples, c=args.opt, bonus=0.0)
                    else:
                        o = np.expand_dims(observation, axis=0)
                        distribution = C51.predict(sess, o)
                        values = C51.Q(distribution)
                action_id = np.random.choice(np.flatnonzero(values == values.max()))
                next_observation, reward, terminal, info = env.step(action_id)
                no = np.expand_dims(next_observation, axis=0)
                next_counts = counts # hack to avoind passing counts
                episode_return.append(reward)
                replay_buffer.add(observation, action_id, reward, terminal,\
                                 counts, next_counts)
                # Training:
                l, _ = C51.train(sess=sess, size=Config.train_size, opt=args.opt, learning_rate = lr)
                if l is not None:
                    C51_loss.append(l)
                    returns[ep, 1] = l
                observation = next_observation
            returns[ep, 0] = discounted_return(episode_return, Config.args.gamma)
            if ep%Config.eval_episode == 0:
                print("Evaluation. Episode ep:%4d, Discounted Return = %g, Epsilon = %g"\
                        %(ep, returns[ep, 0], epsilon))
            if ep%Config.print_episode == 0 and not ep%Config.eval_episode == 0:
                print("Training.  Episode ep:%3d, Discounted Return = %g,Epsilon = %g, Lr = %g, C51 average loss=%g"\
                        %(ep, returns[ep, 0], epsilon, lr, np.mean(C51_loss)))
            if ep% Config.save_episode == 0:
                save_file = {'ep': ep, 'returns': returns, 'episode_data': env.episode_data,\
                        'evaluation_returns': evaluation_returns}
                replay_buffer.save(args.save_name)
                pickle_in = open(args.save_name + '_%d.p'%(ep), 'wb')
                pickle.dump(save_file, pickle_in)
                pickle_in.close()
                saver.save(sess, args.save_name + '.ckpt')

def discounted_return(returns, gamma):
    ret = 0
    for r in reversed(returns):
        ret = r + gamma * ret
    return ret

if __name__ == "__main__":
    args = parser.parse_args()
    args.num_episode += 1
    print('[*] If on server uncomment the saving part')
    #args.save_name = '/next/u/keramati' + args.save_name
    if not os.path.exists(args.save_name + "/summary"):
        os.makedirs(args.save_name + "/summary")
    # Run the egreedy
    egreedy(args)
