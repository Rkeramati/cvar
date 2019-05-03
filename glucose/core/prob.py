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
            self.Logprobsummary = tf.summary.merge_all()

    def _add_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32,\
                shape = [None, self.input_dim], name="input")

    def _build_model(self):
        # Real NVP
        self.nvp = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=np.zeros(num_dims)),
            bijector=tfb.RealNVP(
            num_masked=len(self.config.logprob_layers),
            shift_and_log_scale_fn=tfb.real_nvp_default_template(
            hidden_layers=self.config.logprob_layers)))

    def _add_train_op(self):
        self.loss = -tf.reduce_mean(self.nvp.log_prob(self.input))
        self.loss_batch = -self.nvp.log_prob(self.input)

        tf.summary.scalar("logprb_loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        # Gradient for inference:
        gradient = self.optimizer.compute_gradients(self.loss_batch)
        self.PG = self.lr * tf.math.sqaure(gradient)
        tf.summary.scalar("PG", self.PG)
        self.p_count = 1/(tf.math.exp(self.PG) - 1)
        tf.summary.scalar("P_Count", self.p_count)
        # Train for loss:
        self.train_op = self.optimizer.minimize(self.loss)

    def compute_counts(self, sess, x):
        # X shape : [1, state_size]
        x = np.tile(x, (self.config.nA, 1))
        a = self.config.action_process(self.config.action_map)
        nvp_input = np.concatenate((x, a), axis = 1)
        counts = self.sess.run([self.p_count], feed_dict={self.input: nvp_input})[0]
        return counts

    def train(self, sess, x, a):
        a = self.config.process_action(a)
        nvp_input = np.concatenate((x, a), axis=1)
        self.sess.run([self.train_op], feed_dict={self.x: nvp_input})

