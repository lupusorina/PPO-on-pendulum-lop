from gymnasium.envs.registration import (
    registry,
    register,
    make,
    spec,
)

register(
    id="Pendulum-v1-custom",
    #entry_point="gymnasium_set_state.envs:UpdatedPendulumEnv",
    entry_point="a3_gym_env.envs.pendulum:CustomPendulumEnv",
    reward_threshold=200,
)
