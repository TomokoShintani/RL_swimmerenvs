from gym.envs.registration import register

register(
    id='myenvswimmer-v1',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env:MySwimmerDiffRadii',
)

register(
    id='myenvswimmer-v2',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env_123:MySwimmerDiffRadii123',
)

register(
    id='myenvswimmer-v3',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env_132:MySwimmerDiffRadii132',
)

register(
    id='myenvswimmer-v4',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env_213:MySwimmerDiffRadii213',
)

register(
    id='myenvswimmer-v5',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env_231:MySwimmerDiffRadii231',
)

register(
    id='myenvswimmer-v6',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env_312:MySwimmerDiffRadii312',
)

register(
    id='myenvswimmer-v7',
    entry_point='gym_myswimmer3diffradii.envs.swimmer_env_321:MySwimmerDiffRadii321',
)