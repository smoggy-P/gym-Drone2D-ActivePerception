import numpy as np
import math
from map.grid import OccupancyGridMap
from config import *

class LookAhead(object):
    """Make the drone look at the direction of its velocity

    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        pass

    def plan(self, drone):
        drone.yaw = math.degrees(math.atan2(-drone.velocity[1], drone.velocity[0]))

class Oxford(object):
    """Oxford method to plan gaze

    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        self.last_time_observed_map = OccupancyGridMap(MAP_GRID_SCALE, self.dim)
        self.swep_map = OccupancyGridMap(MAP_GRID_SCALE, self.dim)
    
    # def plan(self, )

    