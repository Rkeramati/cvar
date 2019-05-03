import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class LogProb():
    def __init__(self, config):
        self.config = config
        self.input_dim = self.config.state_size + self.config.action_size
        self.lr = 1e-4
        with tf.variable_scope("logprob"):
            self._add_placeholder()
            self._build_model()
            self._add_train_op()
            self.Logprobsummary = tf.summary.merge_all(scope="logprob")
        with tf.variable_scope("counts"):
            self._add_pg_op()
            self.counts_summary = tf.summary.merge_all(scope="counts")

    def _add_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float64,\
                shape = [None, self.input_dim], name="input")
        self.count_input = tf.placeholder(dtype=tf.float64,\
                shape=[self.config.nA, self.input_dim])
    def _build_model(self):
        # Real NVP
        self.nvp = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=np.zeros(self.input_dim)),
            bijector=tfb.RealNVP(
            num_masked=len(self.config.logprob_layers),
            shift_and_log_scale_fn=tfb.real_nvp_default_template(
            hidden_layers=self.config.logprob_layers)))

    def _add_train_op(self):
        self.loss = -tf.reduce_mean(self.nvp.log_prob(self.input))
        tf.summary.scalar("logprb_loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def _add_pg_op(self):
        self.loss_batch = -self.nvp.log_prob(self.count_input)
        self.dummy_optimizer = tf.train.GradientDescentOptimizer(self.lr)

        # Gradient for inference:
        self.PG = []
        self.p_count = []
        for i in range(self.config.nA):
            gradient = self.dummy_optimizer.compute_gradients(self.loss_batch[i])
            grads, _ = list(zip(*gradient))
            norms = tf.global_norm(grads)
            pg = self.lr * norms**2
            self.PG.append(pg)
            tf.summary.histogram("PG_A%d"%(i), pg)
            p_count = 1/(tf.math.exp(pg) - 1)
            self.p_count.append(p_count)
            tf.summary.scalar("P_Count_A%d"%(i), p_count)

    def compute_counts(self, sess, x):
        # X shape : [1, state_size]
        x = np.tile(x, (self.config.nA, 1))
        a = self.config.action_process(self.config.action_map)
        nvp_input = np.concatenate((x, a), axis = 1)
        counts, summary = sess.run([self.p_count, self.counts_summary],\
                feed_dict={self.count_input: nvp_input})
        return counts, summary

    def train(self, sess, x, a):
        a = self.config.action_process(a)
        nvp_input = np.concatenate((x, a), axis=1)
        _, summary = sess.run([self.train_op, self.Logprobsummary], feed_dict={self.input: nvp_input})
        return summary
