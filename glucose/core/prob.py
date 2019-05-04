import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
'''
    Class for the denisty model,
    input -- config file
    Type -- Real NVP density model with base Gaussian Distribution
'''
class LogProb():
    def __init__(self, config):
        self.config = config
        # Input dimension: state_dim + action_dim pair of (s,a)
        self.input_dim = self.config.state_size + self.config.action_size
        self.lr = 1e-4
        # Building the log prob model
        with tf.variable_scope("logprob"):
            self._add_placeholder()
            self._build_model()
            self._add_train_op()
            self.Logprobsummary = tf.summary.merge_all(scope="logprob")
        # Building pesudo count model
        # Different scope is necessary for calling the summary when different
        # call to loss happens
        with tf.variable_scope("counts"):
            self._add_pg_op()
            self.counts_summary = tf.summary.merge_all(scope="counts")

    def _add_placeholder(self):
        '''
            input -- [Batch size, input_dim]: Input to take a gradient step for realNVP
            count_input -- [nA, input_dim]: Input to compute the gradient and the pesudo count
                           batch size = nA, beucase we call for all batches
        '''
        self.input = tf.placeholder(dtype=tf.float64,\
                shape = [None, self.input_dim], name="input")
        self.count_input = tf.placeholder(dtype=tf.float64,\
                shape=[self.config.nA, self.input_dim])
    def _build_model(self):
        '''
            Build the real NVP model
            Arguments:
                -- Base Distribution: Gaussian
                -- hiden layers: self.config.logprob_layers = list of layer sizes e.g. [32, 32, 32]
        '''
        self.nvp = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=np.zeros(self.input_dim)),
            bijector=tfb.RealNVP(
            num_masked=len(self.config.logprob_layers),
            shift_and_log_scale_fn=tfb.real_nvp_default_template(
            hidden_layers=self.config.logprob_layers)))

    def _add_train_op(self):
        # Training op for taking the gradient step
        self.loss = -tf.reduce_mean(self.nvp.log_prob(self.input))
        tf.summary.scalar("logprb_loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def _add_pg_op(self):
        '''
            Compute the counts:
                Input -- self.count_input = [nA, input_dim]
            This builds the op, for computing the pg = lr * (d(loss)/d(theta))^2 for each actions
            Then the pesduo coun is (exp(PG) - 1)^-1
        '''
        # Not taking the average over batches:
        self.loss_batch = -self.nvp.log_prob(self.count_input)
        # Optimizer for computing the gradients
        self.dummy_optimizer = tf.train.GradientDescentOptimizer(self.lr)

        self.PG = []
        self.p_count = []
        for i in range(self.config.nA):
            gradient = self.dummy_optimizer.compute_gradients(self.loss_batch[i])
            grads, _ = list(zip(*gradient))
            norms = tf.global_norm(grads)
            tf.summary.scalar("PG_norm_%d"%(i), norms)
            pg = self.lr * norms**2
            self.PG.append(pg)
            tf.summary.scalar("PG_A%d"%(i), pg)
            p_count = 1/(tf.math.exp(pg) - 1 + 0.001)
            self.p_count.append(p_count)
            tf.summary.scalar("P_Count_A%d"%(i), p_count)

    def compute_counts(self, sess, x):
        # X shape : [nA, state_dim]
        x = np.tile(x, (self.config.nA, 1))
        a = self.config.action_process(self.config.action_map)
        nvp_input = np.concatenate((x, a), axis = 1) # Input = [x,a] = [batch size=nA, input_dim]
        counts, summary = sess.run([self.p_count, self.counts_summary],\
                feed_dict={self.count_input: nvp_input})
        return counts, summary

    def train(self, sess, x, a):
        # Train on one sample, observed (x,a)
        a = self.config.action_process(a)
        nvp_input = np.concatenate((x, a), axis=1)
        _, summary = sess.run([self.train_op, self.Logprobsummary], feed_dict={self.input: nvp_input})
        return summary
