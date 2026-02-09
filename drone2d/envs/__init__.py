"""Gym environment registration.

Importing this module registers the following environments:

* ``drone-2d-perception-v2`` — :class:`~drone2d.envs.drone_env.Drone2DEnv`
* ``drone-2d-metric-v1``     — :class:`~drone2d.envs.metric_env.MetricEnv`
"""

from gym.envs.registration import register

from drone2d.envs.drone_env import Drone2DEnv
from drone2d.envs.metric_env import MetricEnv

register(id="drone-2d-perception-v2", entry_point="drone2d.envs:Drone2DEnv")
register(id="drone-2d-metric-v1", entry_point="drone2d.envs:MetricEnv")
