import numpy as np

class C51():
    # C51 Class, for tabular setting
    def __init__(self, config, init='uniform', ifCVaR = False, p=None, memory=None):
        '''
        args: init -- cdf initial values,
                'optimistic': put all the mass to the last probability atom = V_max
                'uniform': equal mass for each atom 1/nAtoms
                'random': assign random probability and then normalize to sum to 1

              ifCVaR -- if this class is being used to find the CVaR policy
                It is important in function observe, in order to use argmax of expectation
                or argmax of CVaR to find the target distribution

            config -- config class containing all constants and hyperparameters
        '''

        self.ifCVaR = ifCVaR
        self.config = config
        self.init = init

        if memory is not None:
            self.memory = memory
        # Load:
        if p is not None:
            print("P loaded for c51")
            self.p = p
        # initialize:
        if p is None:
            print("Initailizing P for c51")
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
            else:
                raise Exception("C51: Init type not understood")

        self.dz = (self.config.Vmax - self.config.Vmin)/(self.config.nAtoms-1)
        self.z = np.arange(self.config.nAtoms) * self.dz + self.config.Vmin

    def observe(self, x, a, r, nx, terminal, lr, bonus):
        '''
            Observe the (x, a, r, nx, terminal) and update the distribution
            toward the target distribution
            x -- int, state
            a -- int, action
            r -- float, reward
            nx -- int, next state
            terminal -- bool, if terminal
            lr -- learning rate
            bonus -- amount of bonus given, usually C/sqrt(count)

        '''
        # Choose the optimal action a*
        if not self.ifCVaR: #Normal
            Q_nx = np.sum(self.p[nx, :, :] * self.z, axis=1)
            a_star = np.argmax(Q_nx)
        else: # take the argmax of CVaR
            Q_nx = self.CVaRopt(x, None, self.config.args.alpha, N=self.config.CVaRSamples, bonus=bonus)
            a_star = np.argmax(Q_nx)

        m = np.zeros(self.config.nAtoms) #target distribution

        if not terminal:
            # Apply Optimism:
            cdf = np.cumsum(self.p[nx, a_star, :]) - bonus
            cdf = np.clip(cdf, a_min=0, a_max=1) # Set less than 0 to 0
            cdf[-1] = 1 #set the last to be equal to 1
            cdf[1:] -= cdf[:-1]
            optimistic_pdf = cdf # Set the optimisitc pdf

            # Distribute the probability mass
            tz = np.clip(r + self.config.gamma * self.z,\
                    self.config.Vmin, self.config.Vmax)
            b = (tz - self.config.Vmin)/self.dz
            l = np.floor(b).astype(np.int32); u = np.ceil(b).astype(np.int32)
            idx = np.arange(self.config.nAtoms)

            m[l] += self.p[nx, a_star, idx] * (u-b)
            m[u] += self.p[nx, a_star, idx] * (b-l)

            # taking into account when l == u
            # will be zero for l<b and 1 for l==b
            m[idx] += self.p[nx, a_star, idx] * np.floor((1 + (l-b)))
        # Terminal State:
        else:
            tz = np.clip(r, self.config.Vmin, self.config.Vmax)
            b = (tz - self.config.Vmin)/self.dz
            l = int(np.floor(b)); u = int(np.ceil(b))
            if l == u:
                m[l] = 1
            else:
                m[l] = (u-b)
                m[u] = (b-l)

        # Learning with learning rate
        self.p[x, a, :] = self.p[x, a, :] + lr * (m - self.p[x, a, :])
        # Map back to a probability distribtuion, sum = 1
        self.p[x, a, :] /= np.sum(self.p[x, a, :])

    def train(self, size, lr, counts, opt):
        # Train on "size" samples, opt: optimism constant, counts: visitation count
        if size <= self.memory.count:
            print("warning: not enough samples to train on!! skipped")
            return None
        for _ in range(size):
            x, a, r, nx, terminal = self.memory.sample()
            self.observe(x, a, r, nx, terminal, lr=lr, bonus=opt/np.sqrt(counts[x, a]))
        return None

    def Q(self, x):
        # return the Q values of the state x
        Q_nx = np.sum(self.p[x, :, :] * self.z, axis=1)
        return Q_nx

    def CVaR(self, x, alpha, N=20): #TODO: Vectorize over action space
        '''
            Return the CvaR at level alpha for state x
            args
                x -- int, state
                alpha -- float, risk level
                N -- int, number of samples
        '''
        # Return the CVaR based on Sampling N times
        Q = np.zeros(self.config.nA)
        for a in range(self.config.nA):
            cdf = np.tile(np.cumsum(self.p[x, a, :]), N).reshape(N, self.config.nAtoms)
            values = np.zeros(N)
            tau = np.random.uniform(0, alpha, N).reshape(N, 1)
            idx = np.argmax((cdf > tau) * 1.0, axis = 1)
            values = self.z[idx]
            Q[a] = np.mean(values)
        return Q

    def CVaRopt(self, x, count, alpha, c=0.1, N=20, bonus=None):
        '''
            compute CVaR after a optimisctic shift on the ECDF
            args
                x -- int, state
                count -- array of size nS, nA, number of times x, a gets observed
                alpha -- CVaR risk value
                c -- optimism constant, float
                N -- number of samples to compute the CVaR
        '''
        Q = np.zeros(self.config.nA)
        for a in range(self.config.nA):
            # Apply Optimism
            if count is not None:
                cdf = np.cumsum(self.p[x, a, :]) - c/np.sqrt(count[x, a])
            else:
                cdf = np.cumsum(self.p[x, a, :] - bonus)
                if bonus == None:
                    raise Exception("bonus and count are both None!")
            cdf = np.clip(cdf, a_min=0, a_max=None)
            cdf[-1] = 1 #Match the last one to 1
            # Compute CVaR
            cdf = np.tile(cdf, N).reshape(N, self.config.nAtoms)
            tau = np.random.uniform(0, alpha, N).reshape(N, 1)
            idx = np.argmax((cdf > tau) * 1.0, axis=1)
            # Average
            values = self.z[idx]
            Q[a] = np.mean(values)
            values = np.zeros(N)
        return Q

    def softmax(self, x, temp):
        # Softmax with temprature
        e_x = np.exp((x - np.max(x))/temp)
        return e_x / e_x.sum()
