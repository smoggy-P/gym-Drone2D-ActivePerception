import numpy as np
import math

class LookAhead(object):
    """Make the drone look at the direction of its velocity

    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        pass

    def plan(self, drone):
        drone.yaw = math.degrees(math.atan2(-drone.velocity[1], drone.velocity[0]))

# class Oxford(object):
