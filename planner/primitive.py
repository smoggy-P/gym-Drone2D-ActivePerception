import time
from numpy.linalg import norm
import numpy as np
from config import *
class Waypoint2D(object):
    def __init__(self, pos=np.array([0,0]), vel=np.array([0,0])):
        self.position = pos
        self.velocity = vel

class Trajectory2D(object):
    def __init__(self):
        self.positions = []
        self.velocities = []
    def pop(self):
        self.positions.pop(0)
        self.velocities.pop(0)
    def __len__(self):
        return len(self.positions)

def waypoint_from_traj(coeff, t):
    """Get the waypoint in trajectory with coefficient at time t

    Args:
        coeff (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    waypoint = Waypoint2D()
    waypoint.position = np.around(np.array([1, t, t**2]) @ coeff.T)
    waypoint.velocity = np.array([1, 2*t]) @ coeff[:, 1:].T
    return waypoint

class Primitive(object):
    class Node:
        def __init__(self, pos, vel, cost, parent_index, coeff, itr):
            self.position = pos  
            self.velocity = vel 
            self.cost = cost
            self.parent_index = parent_index
            self.coeff = coeff
            self.itr = itr
            self.get_index()
        def get_index(self):
            self.index = (round(self.position[0])//5, round(self.position[1])//5, round(self.velocity[0])//2, round(self.velocity[1])//2)

    def __init__(self, screen, drone):
        self.u_space = np.arange(-DRONE_MAX_ACC, DRONE_MAX_ACC, 5)
        # self.u_space = np.array([-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15])
        self.dt = 3
        self.sample_num = 30 # sampling number for collision check
        self.target = np.array([drone.x, drone.y])
        self.search_threshold = 20
        self.screen = screen
        self.cost_ratio = 100

    def set_target(self, target_pos):
        self.target = target_pos


    def plan(self, start_pos, start_vel, occupancy_map, agents, update_t):
        """
        A star path search
        input:
            s_x: start x position 
            s_y: start y position 
            gx: goal x position 
            gy: goal y position
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        gx, gy = self.target[0], self.target[1]
        start_node = self.Node(pos=start_pos, 
                               vel=start_vel,
                               cost=0.0, 
                               parent_index=-1,
                               coeff=None,
                               itr=0)

        open_set, closed_set = dict(), dict()
        open_set[start_node.index] = start_node
        itr = 0
        time1 = time.time()
        while 1:
            itr += 1
            if len(open_set) == 0 or itr >= 10:
                # print("No solution found in limitied time")
                goal_node = None
                success = False
                break

            c_id = min(
                open_set,
                key=lambda o: (open_set[o].cost) + norm(open_set[o].position - np.array([gx, gy])))
            current = open_set[c_id]

            if norm(current.position - np.array([gx, gy])) <= self.search_threshold:
                # print("Find goal")
                goal_node = current
                success = True
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            sub_node_list = [self.Node(pos=np.around(np.array([1, self.dt, self.dt**2]) @ np.array([[current.position[0], current.position[1]], [current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]])), 
                                   vel=np.array([1, 2*self.dt]) @ np.array([[current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]]), 
                                   cost=current.cost + (x_acc**2 + y_acc**2)/self.cost_ratio + 1, 
                                   parent_index=current.index,
                                   coeff=np.array([[current.position[0], current.velocity[0], x_acc / 2], [current.position[1], current.velocity[1], y_acc / 2]]),
                                   itr = current.itr + 1) for x_acc in self.u_space 
                                                             for y_acc in self.u_space 
                                                             if self.is_free(np.array([[current.position[0], current.velocity[0], x_acc / 2], [current.position[1], current.velocity[1], y_acc / 2]]), occupancy_map, agents, current.itr) and 
                                                                norm(np.array([1, 2*self.dt]) @ np.array([[current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]])) < DRONE_MAX_SPEED]
            for next_node in sub_node_list:
                if next_node.index in closed_set:
                    continue

                if next_node.index not in open_set:
                    open_set[next_node.index] = next_node  # discovered a new node
                else:
                    if open_set[next_node.index].cost > next_node.cost:
                        # This path is the best until now. record it
                        open_set[next_node.index] = next_node
        print("planning time:", time.time()-time1)
        trajectory = Trajectory2D()
        if success:
            cur_node = goal_node
            
            while(cur_node!=start_node): 
                trajectory.positions.extend([waypoint_from_traj(cur_node.coeff, t).position for t in np.arange(self.dt, 0, -update_t)])
                trajectory.velocities.extend([waypoint_from_traj(cur_node.coeff, t).velocity for t in np.arange(self.dt, 0, -update_t)])
                cur_node = closed_set[cur_node.parent_index]
            trajectory.positions.reverse()
            trajectory.velocities.reverse()

        return trajectory, success

    
    def is_free(self, coeff, occupancy_map, agents, itr):
        """Check if there is collision with the trajectory using sampling method

        Args:
            occupancy_map (_type_): Occupancy Map
            agents (_type_): Dynamic obstacles
        """
        for t in np.arange(0, self.dt, self.dt / self.sample_num):

            position = np.around(np.array([1, t, t**2]) @ coeff.T)

            grid = occupancy_map.get_grid(position[0], position[1])
            if grid == 1:
                return False
            for agent in agents:
                global_t = t + itr * self.dt
                new_position = agent.position + agent.velocity * global_t
                if norm(position - new_position) <= DRONE_RADIUS + agent.radius:
                    # print("collision found with agent in position:", new_position)
                    return False


        return True

