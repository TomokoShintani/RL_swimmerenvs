from gym.envs.registration import register

register(
    id='myswimmergradvis-v3',
    entry_point='gym_myswimmer3gradvis.envs.swimmer_env:MySwimmerGradVis',
)

register(
    id='myswimmergradvis-v2',
    entry_point='gym_myswimmer3gradvis.envs.swimmer_env_k4:MySwimmerGradVis4',
)

register(
    id='myswimmergradvis-v4',
    entry_point='gym_myswimmer3gradvis.envs.swimmer_env_k0_01:MySwimmerGradVisk001',
)