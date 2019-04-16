from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action

import pandas as pd
import numpy as np
import pkg_resources
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, patient_name=None, reward_fun=None, done_fun=None, scenario=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        if reward_fun==None or done_fun==None:
        	raise Exception("Reward and Termina function must be specified")

        seeds = self._seed(seed=seed)
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'adolescent#001'
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=seeds[1])
        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        print(start_time)
        # scenario_seed can be fed to have randomness in sensor only, and fixed meal plan
        # A good simulator should be able to act only given patient's name not random seed
        if scenario is not None:
        	scenario = CustomScenario(start_time=start_time, scenario=scenario)
        else:
        	scenario = RandomScenario(start_time=start_time, seed=seeds[2])
        print(scenario)
        pump = InsulinPump.withName('Insulet')
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        self.reward_fun = reward_fun
        self.done_fun = done_fun

    @staticmethod
    def pick_patient():
        # TODO: cannot be used to pick patient at the env constructing space
        # for now
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            try:
                select = int(input('>>> '))
            except ValueError:
                print('Please input a number.')
                continue

            if select < 1 or select > len(patient_params):
                print('Please input 1 to {}'.format(len(patient_params)))
                continue

            return select

    def _step(self, action):
    	bolus = action[0]; basal = action[1] # decompose both actions
    	act = Action(bolus=bolus, basal=basal)
    	return self.env.step(act, reward_fun=self.reward_fun, done_fun=self.done_fun)

    def _reset(self):
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    def _render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
    	# action 0 : bolus
    	# action 1 : basal
        ub_basal = self.env.pump._params['max_basal']
        lb_basal = self.env.pump._params['min_basal']
        ub_bolus = self.env.pump._params['max_bolus']
        lb_bolus = self.env.pump._params['min_bolus']

        return spaces.Box(low=np.array([lb_bolus, lb_basal]),\
        		 high=np.array([ub_bolus, ub_basal]), shape=None)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))
