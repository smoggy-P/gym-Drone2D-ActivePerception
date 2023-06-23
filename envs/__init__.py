from gym.envs.registration import register
from envs.drone_v2 import Drone2DEnv2

register(
    id="gym-2d-perception-v2",
    entry_point="envs:Drone2DEnv2"
)



