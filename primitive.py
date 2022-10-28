from cmath import sqrt
from numpy.linalg import norm
import numpy as np
from config import *
import time
from scipy import integrate
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
    waypoint.position = np.around(np.array([1, t, t**2]) @ coeff.T)
    waypoint.velocity = np.array([1, 2*t]) @ coeff[:, 1:].T
    return waypoint

def waypoint_to_index(position, velocity):
    return (round(position[0])//5, round(position[1])//5, round(velocity[0])//2, round(velocity[1])//2)

class Primitive(object):
    class Node:
        def __init__(self, pos, vel, cost, idx, parent_index, coeff, itr):
            self.position = pos  
            self.velocity = vel 
            self.cost = cost
            self.index = idx
            self.parent_index = parent_index
            self.coeff = coeff
            self.itr = itr

    def __init__(self, screen):
        self.u_space = np.array([-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15])
        self.dt = 3
        self.sample_num = 10 # sampling number for collision check
        self.target = np.array([0,0])
        self.search_threshold = 10
        self.screen = screen

    def set_target(self, target_pos):
        self.target = target_pos

    def get_successor(self, start_node, occupancy_map, agents):
        """Generate next primitive from start position and check collision

        Args:
            start (int): position of starting point
            occupancy_map (_type_): _description_
        """
        start_position = start_node.position
        start_velocity = start_node.velocity

        suc_node_list = []
        valid_target_num = 0
        for i, x_acc in enumerate(self.u_space):
            for j, y_acc in enumerate(self.u_space):
                coeff = np.array([[start_position[0], start_velocity[0], x_acc / 2], 
                                  [start_position[1], start_velocity[1], y_acc / 2]])
                
                # def func(x):
                #     return math.sqrt((coeff[0,1] + 2*coeff[0,2]*x)**2+(coeff[1,1] + 2*coeff[1,2]*x)**2)
                # edge_cost, _ = integrate.quad(func, 0, self.dt)

                # Collision check
                waypoint = waypoint_from_traj(coeff, self.dt)
                if self.is_free(coeff, occupancy_map, agents, start_node.itr) and norm(waypoint.velocity) < DRONE_MAX_SPEED:
                    valid_target_num += 1
                    suc_node_list.append(self.Node(pos=waypoint.position, 
                                                   vel=waypoint.velocity, 
                                                   cost=start_node.cost + 1, 
                                                   idx=waypoint_to_index(waypoint.position, waypoint.velocity),
                                                   parent_index=start_node.index,
                                                   coeff=coeff,
                                                   itr = start_node.itr + 1))
                #     print("  valid successor, target position:", waypoint_from_traj(coeff, self.dt).position)
                # else:
                #     print("invalid successor, target position:", waypoint_from_traj(coeff, self.dt).position)
        return suc_node_list

    def planning(self, start_pos, start_vel, occupancy_map, agents, update_t):
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
                               idx=waypoint_to_index(start_pos, start_vel),
                               parent_index=-1,
                               coeff=None,
                               itr=0)

        open_set, closed_set = dict(), dict()
        open_set[waypoint_to_index(start_node.position, start_node.velocity)] = start_node
        itr = 0
        while 1:
            itr += 1
            if len(open_set) == 0 or itr >= 10:
                print("No solution found in limitied time")
                goal_node = None
                success = False
                break

            c_id = min(
                open_set,
                key=lambda o: (open_set[o].cost)/2 + norm(open_set[o].position - np.array([gx, gy])) + norm(open_set[o].velocity))
            current = open_set[c_id]

            if norm(current.position - np.array([gx, gy])) <= self.search_threshold:
                print("Find goal")
                goal_node = current
                success = True
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            sub_node_list = self.get_successor(current, occupancy_map, agents)
            for next_node in sub_node_list:
                if next_node.index in closed_set:
                    continue

                if next_node.index not in open_set:
                    open_set[next_node.index] = next_node  # discovered a new node
                else:
                    if open_set[next_node.index].cost > next_node.cost:
                        # This path is the best until now. record it
                        open_set[next_node.index] = next_node
        waypoints = []
        control_points = []
        if success:
            cur_node = goal_node
            
            while(cur_node!=start_node): 
                control_points.extend([waypoint_from_traj(cur_node.coeff, t).position for t in np.arange(self.dt, 0, -update_t)])
                waypoints.append(cur_node.position)
                cur_node = closed_set[cur_node.parent_index]
            control_points.reverse()
        return control_points, waypoints, success

    
    def is_free(self, coeff, occupancy_map, agents, itr):
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
            for agent in agents:
                global_t = t + itr * self.dt
                new_position = agent.position + agent.velocity * global_t
                if norm(wp.position - new_position) <= DRONE_RADIUS + agent.radius:
                    return False


        return True

