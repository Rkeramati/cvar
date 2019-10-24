import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

class HIVEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    np.random.seed(1)
    
    st_pattern = {1: (0.1, 0.01, 0.5), 2: (0.1, 0.05, 0.1),\
            3:(0, 0.01, 0), 4:(0.1, 0.01, 0), 5:(0, 0, 0)}

    self.state_names = ("T1", "T1*", "T2", "T2*", "V", "E")

    self.continuous_dims = np.arange(6)
    self.actions = np.array([[0., 0.], [.7, 0.], [0., .3], [.7, .3]])
    self.nA = 4
    total_time_steps = 1000
    self.episodeCap = 10  #: total of 1000 days
    assert total_time_steps % self.episodeCap == 0, "not dividable actionable steps"
    # Simulation steps
    self.dt = int(total_time_steps/self.episodeCap)  #: measurement every 5 days
    assert self.dt % 5 == 0, "mini step not well defined"
    self.mini_step = int(self.dt/5)

    self.logspace = True  #: whether observed states are in log10 space or not
    # store samples of current episode for drawing
    self.episode_data = np.zeros((7, self.episodeCap + 1)) #saving the data of episode

    if self.logspace:
        self.statespace_limits = np.array([[-5, 8]] * 6)
    else:
        self.statespace_limits = np.array([[0., 1e8]] * 6)
    self.state_space_dims = 6
    self.t = 0
    self.action_space = spaces.Discrete(self.nA)
    self.observation_space = spaces.Box(low=0, high=self.statespace_limits[0, -1], shape=(6, 1), dtype=np.float32)

    # stochasticities
    self.action_noise, self.action_sigma, self.drop_p = st_pattern[3]
    self.drop_value = 0.2 # Drop by 20%

    # Normalizaion for NN
    self.normalize_state = True
    self.normalize_reward = True

  def step(self, a):
    # Simulation
    self.t += 1

    eps1, eps2 = self.actions[a]

    # Addin the noise to actions:
    eps1 = eps1 + np.random.normal(self.action_noise, self.action_sigma)
    eps2 = eps2 + np.random.normal(self.action_noise, self.action_sigma)

    ns = odeint(self.dsdt, self.state, [0, self.dt],
                  args=(eps1, eps2), mxstep=10000)[-1]
    _, _, _, _, V, E = ns
    self.state = ns.copy()

    # the reward function penalizes treatment because of side-effects
    reward = (- 0.1 * V - 2e4 * eps1 ** 2 - 2e3 * eps2 ** 2 + 1e3 * E)
    if self.normalize_reward:
        reward /= 1e5
        #reward = np.sqrt(reward)

    self.state = ns.copy()
    if self.logspace:
        ns = np.log10(ns)

    self.episode_data[:-1, self.t] = self.state
    self.episode_data[-1, self.t - 1] = a
    return self.normalizeState(ns), reward, self.isTerminal(), self.possibleActions()

  def normalizeState(self, s):
    return (s - self.statespace_limits[:, 0])/\
              (self.statespace_limits[:, 1] - self.statespace_limits[:, 0])

  def isTerminal(self):
    return self.t >= self.episodeCap

  def possibleActions(self):
    return np.arange(4)

  def reset(self):
    self.t = 0
    self.episode_data[:] = np.nan
    # non-healthy stable state of the system
    s = np.array([163573., 5., 11945., 46., 63919., 24.])
    self.state = s.copy()
    self.episode_data[:-1, 0] = s
    if self.logspace:
        s = np.log10(s)
    return self.normalizeState(s)

  def dsdt(self, s, t, eps1, eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = .34
    k1 = 8e-7
    k2 = 1e-4
    delta = .7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.
    c = 13.
    rho1 = 1.
    rho2 = 1.
    lambdaE = 1
    bE = 0.3
    Kb = 100
    d_E = 0.25
    Kd = 500
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1. - eps1) * k1 * V * T1
    tmp2 = (1. - f * eps1) * k2 * V * T2
    dT1 = lambda1 - d1 * T1 - tmp1
    dT2 = lambda2 - d2 * T2 - tmp2
    dT1s = tmp1 - delta * T1s - m1 * E * T1s
    dT2s = tmp2 - delta * T2s - m2 * E * T2s
    dV = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
        - ((1. - eps1) * rho1 * k1 * T1 +
            (1. - f * eps1) * rho2 * k2 * T2) * V
    dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
        - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

    return np.array([dT1, dT2, dT1s, dT2s, dV, dE])
