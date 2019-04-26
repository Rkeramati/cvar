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
            if init == 'uniform':
                self.p = np.ones((self.config.nS, self.config.nA,\
                            self.config.nAtoms)) * 1.0/self.config.nAtoms
            elif init == 'random':
                self.p = np.random.rand(self.config.nS, self.config.nA,\
                            self.config.nAtoms)
                # Normalize
                psums = self.p.sum(axis=-1)
                self.p = self.p/psums[:, :, np.newaxis]
            else:
                raise Exception("C51: Init type not understood")

        self.dz = (self.config.Vmax - self.config.Vmin)/(self.config.nAtoms-1)
        self.z = np.arange(self.config.nAtoms) * self.dz + self.config.Vmin
        print("Warning: CVaRopt is geared toward e-greedy, with bonus=0 always!")
    def observe(self, x, a, r, nx, terminal, lr, counts=None):
        '''
            Observe the (x, a, r, nx, terminal) and update the distribution
            toward the target distribution
            x -- int, state
            a -- int, action
            r -- float, reward
            nx -- int, next state
            terminal -- bool, if terminal
            lr -- learning rate
            counts for bonus term

        '''
        '''
            Parallelization Comment:
                Making it fully parallel will be 3.3x faster,
                Making only CVaR part parallel and looping over the Batch
                will be 2.3x faster
                For the matter of readibility I chose the second option.
        '''
        batch_size = x.shape[0]

        # Choose the optimal action a*
        if not self.ifCVaR: #Normal
            Q_nx = np.sum(self.p[nx, :, :] * self.z, axis=-1)
            a_star = np.argmax(Q_nx, axis=-1)
        else: # take the argmax of CVaR
            Q_nx = self.CVaRopt(nx, counts, self.config.args.alpha,\
                    c=self.config.args.opt, N=self.config.CVaRSamples, bonus=0.0)
            a_star = np.argmax(Q_nx, axis=-1)

        m = np.zeros((batch_size, self.config.nAtoms)) #target distribution
        for batch in range(batch_size):
            if not terminal[batch]:
                if counts is not None:
                    # Apply Optimism:
                    cdf = np.cumsum(self.p[nx[batch], a_star[batch], :]) -\
                        self.config.args.opt/np.sqrt(counts[nx[batch], a_star[batch]])
                    cdf = np.clip(cdf, a_min=0, a_max=1) # Set less than 0 to 0
                    cdf[-1] = 1 #set the last to be equal to 1
                    cdf[1:] -= cdf[:-1]
                    optimistic_pdf = cdf # Set the optimisitc pdf
                else:
                    optimistic_pdf = self.p[nx[batch], a_star[batch], :]

                # Distribute the probability mass
                tz = np.clip(r[batch] + self.config.args.gamma * self.z,\
                        self.config.Vmin, self.config.Vmax)
                b = (tz - self.config.Vmin)/self.dz
                l = np.floor(b).astype(np.int32); u = np.ceil(b).astype(np.int32)
                idx = np.arange(self.config.nAtoms)

                m[batch, l] += optimistic_pdf[idx] * (u-b)
                m[batch, u] += optimistic_pdf[idx] * (b-l)

                # taking into account when l == u
                # will be zero for l<b and 1 for l==b
                m[batch, idx] += optimistic_pdf[idx] * np.floor((1 + (l-b)))
            # Terminal State:
            else:
                tz = np.clip(r[batch], self.config.Vmin, self.config.Vmax)
                b = (tz - self.config.Vmin)/self.dz
                l = int(np.floor(b)); u = int(np.ceil(b))
                if l == u:
                    m[batch, l] += 1
                else:
                    m[batch, l] += (u-b)
                    m[batch, u] += (b-l)

            self.p[x[batch], a[batch], :] = self.p[x[batch], a[batch], :] +\
                    lr * (m[batch, :] - self.p[x[batch], a[batch], :])
            # Map back to a probability distribtuion, sum = 1
            self.p[x[batch], a[batch], :] /= np.sum(self.p[x[batch], a[batch], :])

    def train(self, size, lr, counts, opt):
        # Train on "size" samples, opt: optimism constant, counts: visitation count
        if size > self.memory.count:
            if self.memory.count <=1 :
                print("Warning: not enough Smaples! Skipped Training!")
                return None
            size = self.memory.count
            print("warning: Train on all memory")
        x, a, r, nx, terminal = self.memory.sample(size)
        self.observe(x, a, r, nx, terminal, lr=lr, counts=counts)
        return None

    def Q(self, x):
        # return the Q values of the state x
        Q_nx = np.sum(self.p[x, :, :] * self.z, axis=1)
        return Q_nx

    def CVaR(self, x, alpha, N=20):
        '''
            Return the CvaR at level alpha for state x
            args
                x -- int, state
                alpha -- float, risk level
                N -- int, number of samples
            this function only works with 1D input
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
                x -- 2D array of [batch_size, 1]: int, state
                count -- array of size nS, nA, number of times x, a gets observed
                alpha -- CVaR risk value
                c -- optimism constant, float
                N -- number of samples to compute the CVaR
        '''
        '''
            parallelization Comment:
                Since nA is small in the regime we are working
                Best performing parallel will be parallel over states
                not action space, which is almosr 3x faster
        '''

        batch_size = x.shape[0] # Number of bathces, x of

        Q = np.zeros((batch_size, self.config.nA))
        for a in range(self.config.nA):
            # Apply Optimism
            if count is not None:
                cdf = np.cumsum(self.p[x, a, :], axis=-1) \
                        - np.expand_dims(c/np.sqrt(count[x, a]), axis=-1)
            else:
                cdf = np.cumsum(self.p[x, a, :], axis=-1) - np.expand_dims(bonus, axis=-1)
                if bonus is None:
                    raise Exception("bonus and count are both None!")
            cdf = np.clip(cdf, a_min=0, a_max=None)

            cdf[..., -1] = 1 #Match the last one to 1
            # Compute CVaR
            tau = np.expand_dims(np.random.uniform(0, alpha, N).reshape(N, 1), axis = 0)
            cdf = np.expand_dims(cdf, axis = 1)
            idx = np.argmax((cdf > tau) * 1.0, axis=-1)
            # Average
            values = self.z[idx]
            Q[:, a] = np.mean(values, axis=-1)
        return Q

    def softmax(self, x, temp):
        # Softmax with temprature
        e_x = np.exp((x - np.max(x))/temp)
        return e_x / e_x.sum()
