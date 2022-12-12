from gym.envs.registration import register

register(
    id="gym-2d-perception-v0",
    entry_point="gym_2d_perception.envs:Drone2DEnv"
)