from gym.envs.registration import register
from envs.drone_v2 import Drone2DEnv2
from envs.metric_env import MetricEnv

register(
    id="gym-2d-perception-v2",
    entry_point="envs:Drone2DEnv2"
)

register(
    id="gym-metric-v1",
    entry_point="envs:MetricEnv"
)



