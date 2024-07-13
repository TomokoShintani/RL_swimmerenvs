from gym.envs.registration import register

register(
    id='myenvswimmer-v1',
    entry_point='gym_myswimmer3sameradii.envs.swimmer_env:MySwimmerSameRadii',
)