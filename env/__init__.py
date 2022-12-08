from gym.envs.registration import register

register(
    id="gym-2d-perception-v0",
    entry_point="env.gym_2d:Drone2DEnv"
)