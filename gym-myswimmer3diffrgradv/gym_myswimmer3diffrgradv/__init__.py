from gym.envs.registration import register

register(
    id='myswimmerdiffrgradv-v3',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env:MySwimmerDiffRGradV',
)

register(
    id='myswimmerdiffrgradv-v4',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffratio:MySwimmerDiffRGradV56',
)

register(
    id='myswimmerdiffrgradv-v5',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffr321:MySwimmerDiffRGradV321',
)

register(
    id='myswimmerdiffrgradv-v6',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffr213:MySwimmerDiffRGradV213',
)

register(
    id='myswimmerdiffrgradv-v7',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffr312:MySwimmerDiffRGradV312',
)

register(
    id='myswimmerdiffrgradv-v8',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffr123:MySwimmerDiffRGradV123',
)

register(
    id='myswimmerdiffrgradv-v9',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffr132:MySwimmerDiffRGradV132',
)

register(
    id='myswimmerdiffrgradv-v10',
    entry_point='gym_myswimmer3diffrgradv.envs.swimmer_env_diffr231:MySwimmerDiffRGradV231',
)