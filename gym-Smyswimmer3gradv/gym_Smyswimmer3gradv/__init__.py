from gym.envs.registration import register

register(
    id='myswimmergradvS-v0',
    entry_point='gym_Smyswimmer3gradv.envs.swimmer_envS:MySwimmerGradVS',
)