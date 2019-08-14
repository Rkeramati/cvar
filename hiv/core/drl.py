import numpy as np
import tensorflow as tf

class C51():
    # C51 Class, for Function Approximation Setting
    def __init__(self, config, memory, ifCVaR = False):
        '''
        args: ifCVaR -- if this class is being used to find the CVaR policy
                        It is important in function get_target, in order to use argmax of expectation
                        or argmax of CVaR to find the target distribution
              memory -- Class of replay as a replay memory, either empty or loaded already
              config -- config class containing all constants and hyperparameters

              Architecture:
                  Input -> Dense x [X] -> List of size nA, each size nAtoms
                  self.config.num_layers, self.config.layer_size
        '''
        self.ifCVaR = ifCVaR
        self.config = config
        self.memory = memory
        # Atoms:
        self.dz = (self.config.Vmax - self.config.Vmin)/(self.config.nAtoms-1)
        self.z = np.arange(self.config.nAtoms) * self.dz + self.config.Vmin

        # Building tf model
        with tf.variable_scope("C51"):
            self.add_placeholder()
            self.build_model()
            self.add_optimizer()
            self.C51summary = tf.summary.merge_all(scope="C51")

    def add_placeholder(self):
        '''
            Adding tf placeholders,
                target_distribution, target mask -- list of size nA of tf placeholders
        '''
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.config.state_size],\
                name="state_placeholder")
        self.target_distribution = [tf.placeholder(dtype=tf.float32, shape=[None, self.config.nAtoms],\
                name="target_distribution_%d"%(i)) for i in range(self.config.nA)]
        self.target_mask = [tf.placeholder(dtype=tf.float32, shape=[None],\
                name="mask_%d"%(i)) for i in range(self.config.nA)]
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

    def build_model(self):
        '''
            Desne model"" state_size -> hidden_layer 1 -> hidden_layer 2 --> [nAtoms] of size nA
        '''
        out = self.x
        for layer in range(self.config.num_layers):
            out = tf.layers.dense(out, units=self.config.layer_size[layer],\
                    activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform(),\
                    name="dense_%d"%(layer))
            tf.summary.histogram(out.op.name, out)
        self.distribution = []
        for i in range(self.config.nA):
            layer = tf.layers.dense(out, units=self.config.nAtoms,\
                    activation=tf.nn.softmax, kernel_initializer=tf.initializers.glorot_uniform(),\
                    name = "output_%d"%(i))
            tf.summary.histogram(layer.op.name, layer)
            self.distribution.append(layer)

    def add_optimizer(self):
        '''
            Adam optimizer, loss = cross entropy
            target mask -- targetmask[a][i] = 1 if batch i was generated by taking action a
        '''
        self.loss = 0
        for i in range(self.config.nA):
            # Only train on the taken action: i.e mask = 1
            self.loss += tf.reduce_sum(self.target_distribution[i] * tf.log(self.distribution[i]),\
                    axis=-1) * self.target_mask[i]
        self.loss = -tf.reduce_mean(self.loss, axis=-1)
        tf.summary.scalar("loss", self.loss)
        # TODO: Maybe add learning rate decay
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="Adam")
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        grads, _ = list(zip(*grads_and_vars))
        norms = tf.global_norm(grads)
        tf.summary.scalar('gradient_norm', norms)
        # TODO: Maybe add gradient clipping
        self.train_op = self.optimizer.apply_gradients(grads_and_vars)

    def predict(self, sess, state):
        # input satate = [batch, state_size]
        # Get the distribtuion (pdf) of the input_state
        out = sess.run(self.distribution, feed_dict={self.x: state})
        return out

    def CVaRopt(self, distribution, count, alpha, c, N=20, bonus=None):
        '''
            compute CVaR after a optimisctic shift on the ECDF
            args
                distribution -- list of length nA, elements = (B, nAtoms)
                count -- np.array (B, nA)
                alpha -- CVaR risk value
                c -- optimism constant, float
                N -- number of samples to compute the CVaR
                bonus -- in case for fix bounes
                    For computin CVaR, pass None to count and 0 to bonus, or pass c=0
        '''
        '''
            parallelization Comment:
                Since nA is small in the regime we are working
                Best performing parallel will be parallel over states
                not action space, which is almosr 3x faster
        '''

        batch_size = distribution[0].shape[0] # Number of bathces

        Q = np.zeros((batch_size, self.config.nA))

        for a in range(self.config.nA):
            # Apply Optimism
            if count is not None:
                cdf = np.cumsum(distribution[a], axis=-1) \
                        - np.expand_dims(c/np.sqrt(count[:, a] + 0.001), axis=-1)
            else:
                cdf = np.cumsum(distribution[a], axis=-1) - np.expand_dims(bonus, axis=-1)
                if bonus is None:
                    raise Exception("bonus and count are both None!")

            cdf = np.clip(cdf, a_min=0, a_max=None)

            cdf[..., -1] = 1 #Match the last one to 1
            # Compute CVaR
            tau = np.expand_dims(np.random.uniform(0, alpha, N).reshape(N, 1), axis = 0)
            cdf = np.expand_dims(cdf, axis = 1)
            idx = np.argmax((cdf > tau) * 1.0, axis=-1) # argmax returns the last max
            # Average
            values = self.z[idx]
            Q[:, a] = np.mean(values, axis=-1)
        return Q

    def get_target(self, x, a, r, nx, terminal, next_counts, sess):
        '''
            Compute the target distribution of (x, a, r, nx, terminal)
            x -- state: shape [Batch Size, state_dim]
            a -- action: shape [Batch Size]
            r -- reward: shape [Batch Size]
            nx -- next state: shape [Batch Size, state_dim]
            terminal -- bool terminal: shape [Batch Size]
            next_counts -- Pesudo Count for the next_State, [Batch size, nA]
            sess -- tf session

            Comment: For e_greedy opt = 0, no shift will apply: This will be checked by args
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
        next_distribution = self.predict(sess, nx)

        if not self.ifCVaR: # Normal Expectation
            Q_nx = np.zeros((batch_size, self.config.nA))
            for action in range(self.config.nA):
                Q_nx[:, action] = np.sum(next_distribution[action] * self.z, axis=-1)
            a_star = np.argmax(np.array(Q_nx), axis=-1)
        else: # take the argmax of CVaR
            Q_nx = self.CVaRopt(next_distribution, next_counts, self.config.args.alpha,\
                    c=self.config.args.opt, N=self.config.CVaRSamples, bonus=0.0)
            # TODO: Randomize per row the argmax selection
            a_star = np.argmax(Q_nx, axis=-1)

        # Target Distribution: List of size nA, each [Batch Size, nAtoms]
        m = [np.zeros((batch_size, self.config.nAtoms)) for i in range(self.config.nA)]
        # Mask: List of size nA each size batch size, if mask[a][i] = 1 means action a was taken in batch i
        mask = [np.zeros(batch_size) for i in range(self.config.nA)]
        for batch in range(batch_size):
            # Setting the mask
            mask[a[batch]][batch] = 1
            # setting the target distribution
            if not terminal[batch]:
                # Apply Optimism:
                cdf = np.cumsum(next_distribution[a_star[batch]][batch, :], axis=-1) -\
                        self.config.args.opt/np.sqrt(next_counts[batch, a_star[batch]] + 0.001)
                cdf = np.clip(cdf, a_min=0, a_max=1) # Set less than 0 to 0
                cdf[-1] = 1 #set the last to be equal to 1
                cdf[1:] -= cdf[:-1]
                optimistic_pdf = cdf # Set the optimisitc pdf

                # Distribute the probability mass
                tz = np.clip(r[batch] + self.config.args.gamma * self.z,\
                        self.config.Vmin, self.config.Vmax)
                b = (tz - self.config.Vmin)/self.dz
                l = np.floor(b).astype(np.int32); u = np.ceil(b).astype(np.int32)
                idx = np.arange(self.config.nAtoms)

                m[a[batch]][batch, l] += optimistic_pdf[idx] * (u-b)
                m[a[batch]][batch, u] += optimistic_pdf[idx] * (b-l)

                # taking into account when l == u
                # will be zero for l<b and 1 for l==b
                m[a[batch]][batch, idx] += optimistic_pdf[idx] * np.floor((1 + (l-b)))
            # Terminal State:
            else:
                tz = np.clip(r[batch], self.config.Vmin, self.config.Vmax)
                b = (tz - self.config.Vmin)/self.dz
                l = int(np.floor(b)); u = int(np.ceil(b))
                if l == u:
                    m[a[batch]][batch, l] += 1
                else:
                    m[a[batch]][batch, l] += (u-b)
                    m[a[batch]][batch, u] += (b-l)
        return m, mask

    def train(self, sess, size, opt, learning_rate):


        # Train on "size" samples, opt: optimism constant
        if size > self.memory.count:
            if self.memory.count <=1 :
                print("Warning: not enough Smaples! Skipped Training!")
                return None, None
            size = self.memory.count
            # print("warning: Train on all memory")
        x, a, r, nx, terminal, next_counts = self.memory.sample(size)
        target_dist, target_mask = self.get_target(x, a, r, nx, terminal, next_counts, sess)

        # Dictionary comprehension for feed_dict, since we have a list of tf.placeholder
        dic = {i: d for i, d in zip(self.target_distribution, target_dist)}
        dic2 = {i: d for i, d in zip(self.target_mask, target_mask)}
        dic.update(dic2)
        dic3 = {self.x: x, self.learning_rate: learning_rate}
        dic.update(dic3)

        # Train
        _, loss = sess.run([self.train_op, self.loss], feed_dict=dic)
        return loss, None

    def Q(self, distribution):
        batch_size = distribution[0].shape[0]
        Q = np.zeros((batch_size, self.config.nA))
        for action in range(self.config.nA):
            Q[:, action] = np.sum(distribution[action] * self.z, axis=-1)
        return Q
