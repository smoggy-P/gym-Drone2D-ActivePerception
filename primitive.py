import numpy as np

class Waypoint2D(object):
    def __init__(self, pos=np.array([0,0]), vel=np.array([0,0])):
        self.position = pos
        self.velocity = vel

def waypoint_from_traj(coeff, t):
    """Get the waypoint in trajectory with coefficient at time t

    Args:
        coeff (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    waypoint = Waypoint2D()
    waypoint.position = np.array([1, t, t**2]) @ coeff.T
    waypoint.velocity = np.array([1, 2*t]) @ coeff[:, 1:].T
    return waypoint

class Primitive(object):
    def __init__(self):
        self.u_space = np.array([-1, 0, 1])
        self.dt = 1
        self.sample_num = 10 # sampling number for collision check

    def get_successor(self, start_position, start_velocity, occupancy_map, agents):
        """Generate next primitive from start position and check collision

        Args:
            start (int): position of starting point
            occupancy_map (_type_): _description_
        """
        successor_list = []
        coeff_list = []
        for x_acc in self.u_space:
            for y_acc in self.u_space:
                coeff = np.array([[start_position[0], start_velocity[0], x_acc / 2], 
                                  [start_position[1], start_velocity[1], y_acc / 2]])
                
                # Collision check
                if self.is_free(coeff, occupancy_map, agents):
                    coeff_list.append(coeff) # x(t) = a1 + b1 * t + c1 * t^2
                    successor_list.append(waypoint_from_traj(coeff, self.dt))
                    print("  valid successor, target position:", waypoint_from_traj(coeff, self.dt).position)
                else:
                    print("invalid successor, target position:", waypoint_from_traj(coeff, self.dt).position)
        return coeff_list, successor_list

    

    def is_free(self, coeff, occupancy_map, agents):
        """Check if there is collision with the trajectory using sampling method

        Args:
            occupancy_map (_type_): Occupancy Map
            agents (_type_): Dynamic obstacles
        """
        for t in np.arange(0, self.dt, self.dt / self.sample_num):
            wp = waypoint_from_traj(coeff, t)
            grid = occupancy_map.get_grid(wp.position[0], wp.position[1])
            if grid == 1 or grid == 3:
                return False
        return True


if __name__ == '__main__': 
    w = Primitive()
    a = w.get_successor(start_position=np.array([0,0]), start_velocity=np.array([0,0]))