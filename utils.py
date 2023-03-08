import numpy as np

grid_type = {
    'DYNAMIC_OCCUPIED' : 3,
    'OCCUPIED' : 1,
    'UNOCCUPIED' : 2,
    'UNEXPLORED' : 0
}

class Waypoint2D(object):
    def __init__(self, pos=np.array([0,0]), vel=np.array([0,0])):
        self.position = pos
        self.velocity = vel

class Trajectory2D(object):
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.accelerations = []
    def pop(self):
        self.positions.pop(0)
        self.velocities.pop(0)
        self.accelerations.pop(0)
    def clear(self):
        self.positions = []
        self.velocities = []
        self.accelerations = []
    def __len__(self):
        return len(self.positions)
    def get_swep_map(self, swep_map):
        for i, pos in enumerate(self.positions):
            # unparam
            swep_map[int(pos[0]//10), int(pos[1]//10)] = i * 0.1