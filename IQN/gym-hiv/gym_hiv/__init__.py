from gym.envs.registration import register

register(
    id='hiv-v0',
    entry_point='gym_hiv.envs:HIVEnv',
)
