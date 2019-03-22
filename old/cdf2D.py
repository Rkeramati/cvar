import numpy as np
import matplotlib.pyplot as plt
# import gym
# from env import Gridworld
#from mrp import machine_repair
#from cliff import Cliff
from terrain import Nav2D

import argparse
parser = argparse.ArgumentParser(description='take input')
parser.add_argument('name', help='an integer for the accumulator')
parser.add_argument('trial', help='trial numbers')
parser.add_argument('opt', help='optimism count')

class Config():
    def __init__(self, nS, nA):
        self.Vmin = -0.1
        self.Vmax = 1.1
        self.nAtoms = 51
        self.nS = nS
        self.nA = nA
        self.gamma = 0.99
        self.max_e = 0.9
        self.min_e = 0.1
        self.max_alpha = 0.9
        self.min_alpha = 0.1
        self.episode_ratio = 2

class C51():
    def __init__(self, config, init='uniform', ifCVaR = False, temp=10):
        self.ifCVaR = ifCVaR
        self.config = config
        self.init = init
        self.temp = temp

        if init == 'optimistic':
            self.p = np.zeros((self.config.nS, self.config.nA,\
                        self.config.nAtoms))
            self.p[:, :, -2] = 1
        elif init == 'uniform':
            self.p = np.ones((self.config.nS, self.config.nA,\
                        self.config.nAtoms)) * 1.0/self.config.nAtoms
        elif init == 'random':
            self.p = np.random.rand(self.config.nS, self.config.nA,\
                        self.config.nAtoms)
            for x in range(self.config.nS):
                for a in range(self.config.nA):
                    self.p[x, a, :] /= np.sum(self.p[x, a, :])
        elif init =='temp':
            self.p = np.random.rand(self.config.nS, self.config.nA,\
                        self.config.nAtoms)
            for x in range(self.config.nS):
                for a in range(self.config.nA):
                    self.p[x, a, :] = self.softmax(self.p[x, a, :], temp=temp)
        else:
            self.p = np.ones((self.config.nS, self.config.nA,\
                        self.config.nAtoms)) * 1.0/self.config.nAtoms
            self.p[:, :, 0] = 10
            for x in range(self.config.nS):
                for a in range(self.config.nA):
                    self.p[x, a, :] /= np.sum(self.p[x, a, :])

        self.dz = (self.config.Vmax - self.config.Vmin)/(self.config.nAtoms-1)
        self.z = np.arange(self.config.nAtoms) * self.dz + self.config.Vmin
    def observe(self, x, a, r, nx, terminal, alpha, bonus):
        if not self.ifCVaR: #Normal
            Q_nx = np.zeros(self.config.nA)
            for at in range(self.config.nA):
                Q_nx[at] = np.sum(self.p[nx, at, :] * self.z)
            a_star = np.argmax(Q_nx)
        else:
            Q_nx = self.CVaR(x, alpha, N=20)
            a_star = np.argmax(Q_nx)

        m = np.zeros(self.config.nAtoms)

        if not terminal:
            pdf = self.p[nx, a_star, :]
            cdf = np.cumsum(pdf)
            cdf -= bonus
            for i in range(cdf.shape[0]):
                if cdf[i] <= 0:
                    cdf[i] = 0
            cdf[-1] = 1
            cdf[1:] -= cdf[:-1]
            optimistic_pdf = cdf

            for i in range(self.config.nAtoms):
                tz = np.clip(r + self.config.gamma*self.z[i],\
                        self.config.Vmin, self.config.Vmax)
                b = (tz - self.config.Vmin)/self.dz
                l = int(np.floor(b)); u = int(np.ceil(b))
                if l == u:
                    m[l] += optimistic_pdf[i]
                else:
                    m[l] += optimistic_pdf[i] * (u-b)
                    m[u] += optimistic_pdf[i] * (b-l)
        else:
            tz = np.clip(r, self.config.Vmin, self.config.Vmax)
            b = (tz - self.config.Vmin)/self.dz
            l = int(np.floor(b)); u = int(np.ceil(b))
            if l == u:
                m[l] = 1
            else:
                m[l] = (u-b)
                m[u] = (b-l)

        self.p[x, a, :] = self.p[x, a, :] + alpha * (m - self.p[x, a, :])
        if self.init == 'temp':
            self.p[x, a, :] = self.softmax(self.p[x, a, :], self.temp)
        else:
            self.p[x, a, :] /= np.sum(self.p[x, a, :])

    def Q(self, x):
        Q_nx = np.zeros(self.config.nA)
        for at in range(self.config.nA):
            Q_nx[at] = np.sum(self.p[x, at, :] * self.z)
        return Q_nx

    def CVaR(self, x, alpha, N=20):
        Q = np.zeros(self.config.nA)
        for a in range(self.config.nA):
            values = np.zeros(N)
            for n in range(N):
                tau = np.random.uniform(0, alpha)
                idx = np.argmax((np.cumsum(self.p[x, a, :]) > tau) * 1.0)
                z = self.z[idx]
                values[n] = z
            Q[a] = np.mean(values)
        return Q
    def CVaRopt(self, x, count, alpha, c=0.1, N=20):
        Q = np.zeros(self.config.nA)
        for a in range(self.config.nA):
            values = np.zeros(N)
            for n in range(N):
                cdf = np.cumsum(self.p[x, a, :])
                cdf = cdf - c/np.sqrt(count[x, a])
                for i in range(cdf.shape[0]):
                    if cdf[i] < 0:
                        cdf[i] = 0
                total_so_far = cdf[-1]
                cdf[-1] += (1-total_so_far)

                tau = np.random.uniform(0, alpha)
                idx = np.argmax(((cdf) > tau) * 1.0)
                z = self.z[idx]
                values[n] = z
            Q[a] = np.mean(values)
        return Q
    def softmax(self, x, temp):
        e_x = np.exp((x - np.max(x))/temp)
        return e_x / e_x.sum()

def main(name, version, opt, opt_backup):
    world = Nav2D()

    config = Config(world.nS, world.nA)
    config.Vmin = -150; config.Vmax = 0

    c51 = C51(config, init = 'random', ifCVaR = True)
    c51_eval = C51(config, init = 'random', ifCVaR = True)

    counts = np.zeros((world.nS, world.nA)) + 1

    num_episode = 100000
    trial = 20
    returns = np.zeros((num_episode, trial))
    returns_online = np.zeros((num_episode, trial))

    CVaRs = np.zeros((num_episode, world.nS, world.nA))
    CVaRsopt = np.zeros((num_episode, world.nS, world.nA))
    for ep in range(num_episode):
        terminal = False
        alpha = max(0.01, 0.5 + ep * ((0.01 - 0.5)/(num_episode/4)))
        o = world.reset()
        o_init = o
        ret = 0
        while not terminal:
            values = c51.CVaRopt(o, counts, c=opt, alpha=0.25, N=50)
            a = np.random.choice(np.flatnonzero(values == values.max()))
            no, r, terminal = world.step(a)
            counts[o, a] += 1
            ret += r
            c51.observe(o, a, r, no, terminal, alpha, bonus=opt/np.sqrt(counts[o, a]))
            c51_eval.observe(o, a, r, no, terminal, alpha, bonus=0)
            o = no
        if ep%50 == 0:
            # Evaluation
            tot_rep = np.zeros(trial)

            values = np.zeros((world.nS, world.nA))
            for states in range(world.nS):
                values[states, :] = c51_eval.CVaR(states, alpha=0.25, N=50)

            for ep_t in range(trial):
                terminal = False
                o = world.reset()
                o_init = o
                ret = 0
                step = 0
                while not terminal and step <= 200:
                    #values = c51_eval.CVaR(o, alpha=0.25, N=50)
                    a = np.random.choice(np.flatnonzero(values[o, :] == values[o, :].max()))
                    no, r, terminal = world.step(a)
                    ret += r
                    o = no
                    step += 1
                tot_rep[ep_t] = ret
            returns[ep, :] = tot_rep
            # Online:
            tot_rep = np.zeros(trial)

            values = np.zeros((world.nS, world.nA))
            for states in range(world.nS):
                values[states, :] = c51.CVaRopt(states, counts, c=opt, alpha=0.25, N=50)

            for ep_t in range(trial):
                terminal = False
                o = world.reset()
                o_init = o
                ret = 0
                step = 0
                while not terminal and step <= 200:
                    #values = c51.CVaRopt(o, counts, c=opt, alpha=0.25, N=50)
                    a = np.random.choice(np.flatnonzero(values[o, :] == values[o, :].max()))
                    no, r, terminal = world.step(a)
                    ret += r
                    o = no
                    step += 1
                tot_rep[ep_t] = ret
            returns_online[ep, :] = tot_rep
            if ep%100 == 0:
                print('episode: %d'%(ep))
            np.save(name + '_cdf_online_%d.npy'%(version), returns_online)
            np.save(name + '_cdf_eval_%d.npy'%(version), returns)
        if ep%2000 == 0:
            np.save(name + '_c51_p_%d.npy'%(ep), c51.p)
            np.save(name + '_c51_counts_%d.npy'%(ep), counts)
            np.save(name + '_c51_eval_p_%d.npy'%(ep), c51_eval.p)
if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(int(args.trial)):
        print('Trail: %d out of %d'%(i, int(args.trial)))
        main(args.name, i, float(args.opt), 0)
