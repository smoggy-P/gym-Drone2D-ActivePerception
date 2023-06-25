
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from math import cos, sin, sqrt, atan2, radians
from sklearn.cluster import DBSCAN
from cvxpy.error import SolverError
from utils import Trajectory2D, Waypoint2D, grid_type
from matplotlib.patches import Circle, Ellipse
from utils import *
import sys
sys.path.insert(0, '/home/cc/moji_ws/forces_pro_client/')  # On Windows, note the doubly-escaped backslashes
import forcespro

class Planner:
    def __init__(self, drone, params):
        self.trajectory = Trajectory2D()
        self.params = params
        self.target = np.array([drone.x, drone.y, 0, 0])

    def set_target(self, target):
        self.target = np.zeros(4)
        self.target[:2] = target

    def is_free(self, position, t, occupancy_map, trackers):
        if np.isnan(position).any():
            return False
        
        safe_distance = self.params.drone_radius + 10

        grid = occupancy_map.get_grid(position[0] - safe_distance, position[1])
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0], position[1])
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0] + safe_distance, position[1])
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0], position[1] - safe_distance)
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0], position[1] + safe_distance)
        if grid == 1:
            return False

        for tracker in trackers:
            if tracker.active:
                new_position = tracker.estimate_pos(t)
                if norm(position - new_position) <= self.params.drone_radius + self.params.agent_radius + 5 + self.params.var_cam:
                    return False
        return True

    def plan(self, drone   : Drone2D,
                   update_t: float):
        raise NotImplementedError("No planner implemented!")

    def replan_check(self, drone):
        raise NotImplementedError("No replan checker implemented!")

class NoMove(Planner):

    def plan(self, drone: Drone2D,
                   dt   : float):
        self.target = np.array([-1, -1, 0, 0])
        return True
    
    def replan_check(self, drone):
        return False, drone.map.grid_map

class Primitive(Planner):

    class Primitive_Node:
        def __init__(self, pos, vel, cost, target, parent_index, coeff, itr):
            self.position = pos  
            self.velocity = vel 
            self.cost = cost
            self.parent_index = parent_index
            self.coeff = coeff
            self.itr = itr
            self.total_cost = cost + 0.5*norm(pos-target) + 0.1*norm(vel)
            self.get_index()
        def __lt__(self, other_node):
            return self.total_cost < other_node.total_cost
        def get_index(self):
            self.index = (round(self.position[0])//10, round(self.position[1])//10, round(self.velocity[0]), round(self.velocity[1]))
    
    def __init__(self, drone, params):
        super(Primitive, self).__init__(drone, params)
        
        if params.drone_max_speed <= 40: 
            self.u_space = np.arange(-params.drone_max_acceleration, params.drone_max_acceleration, 0.4 * params.drone_max_speed - 5)
        else:
            self.u_space = np.arange(-params.drone_max_acceleration, params.drone_max_acceleration, 4)

        self.dt = 2
        self.sample_num = params.drone_max_speed * self.dt // params.map_scale # sampling number for collision check
        self.target = np.array([drone.x, drone.y, 0, 0])
        self.search_threshold = 10
        self.phi = 10

    @classmethod
    def waypoint_from_traj(self, coeff, t):
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

    def plan(self, drone   : Drone2D,
                   update_t: float):
        
        if len(self.trajectory) != 0:
            return True
        
        start_pos = np.array([drone.x, drone.y])
        start_vel = drone.velocity
        occupancy_map = drone.map
        
        self.trajectory = Trajectory2D()
        start_node = self.Primitive_Node(pos=start_pos, 
                                        vel=start_vel,
                                        cost=0, 
                                        target=self.target[:2],
                                        parent_index=-1,
                                        coeff=None,
                                        itr=0)

        open_set, closed_set = dict(), dict()
        open_set[start_node.index] = start_node
        itr = 0
        while 1:
            itr += 1
            if len(open_set) == 0 or itr >= 100:
                # print("No solution found in limitied time")
                goal_node = None
                success = False
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].total_cost)
            current = open_set[c_id]

            if norm(current.position - self.target[:2]) <= self.search_threshold:
                # print("Find goal")
                goal_node = current
                success = True
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            sub_node_list = []
            for x_acc in self.u_space:
                for y_acc in self.u_space: 
                    if norm(np.array([1, 2*self.dt]) @ np.array([[current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]])) < self.params.drone_max_speed:
                        coeff = np.array([[current.position[0], current.velocity[0], x_acc / 2], [current.position[1], current.velocity[1], y_acc / 2]])
                        add_successor = True
                        
                        for t in np.arange(0, self.dt, self.dt / self.sample_num):
                            position = np.around(np.array([1, t, t**2]) @ coeff.T)
                            global_t = t + current.itr * self.dt
                            if not self.is_free(position, global_t, occupancy_map, drone.trackers):
                                add_successor = False
                                break
                        
                        if add_successor:
                            successor = self.Primitive_Node(pos=np.around(np.array([1, self.dt, self.dt**2]) @ np.array([[current.position[0], current.position[1]], [current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]])), 
                                                            vel=np.array([1, 2*self.dt]) @ np.array([[current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]]), 
                                                            cost=current.cost + (x_acc**2 + y_acc**2)/100 + 10, 
                                                            target=self.target[:2],
                                                            parent_index=current.index,
                                                            coeff=np.array([[current.position[0], current.velocity[0], x_acc / 2], [current.position[1], current.velocity[1], y_acc / 2]]),
                                                            itr = current.itr + 1)
                            sub_node_list.append(successor)

            for next_node in sub_node_list:
                if next_node.index in closed_set:
                    continue

                if next_node.index not in open_set:
                    open_set[next_node.index] = next_node  # discovered a new node
                else:
                    if open_set[next_node.index].cost > next_node.cost:
                        # This path is the best until now. record it
                        open_set[next_node.index] = next_node
        # print("planning time:", time.time()-time1)
        if success:
            cur_node = goal_node
            
            while(cur_node!=start_node): 
                self.trajectory.positions.extend([self.waypoint_from_traj(cur_node.coeff, t).position for t in np.arange(self.dt, 0, -update_t)])
                self.trajectory.velocities.extend([self.waypoint_from_traj(cur_node.coeff, t).velocity for t in np.arange(self.dt, 0, -update_t)])
                self.trajectory.accelerations.extend([np.array([0, 0]) for t in np.arange(self.dt, 0, -update_t)])
                cur_node = closed_set[cur_node.parent_index]
            self.trajectory.positions.reverse()
            self.trajectory.velocities.reverse()
        return success

    def replan_check(self, drone):
        occupancy_map = drone.map.grid_map
        swep_map = np.zeros_like(occupancy_map)
        for i, pos in enumerate(self.trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.params.dt
            for tracker in drone.trackers:
                if tracker.active:
                    if norm(tracker.estimate_pos(i * self.params.dt) - pos) <= self.params.drone_radius + self.params.agent_radius:
                        self.trajectory.clear()
                        return True, swep_map
        if np.sum(np.where((occupancy_map==1),1, 0) * swep_map) > 0:
            self.trajectory.clear()
            return True, swep_map
        return False, swep_map
    
class MPC(Planner):
    
    def __init__(self, drone, params):
        self.params = params
        dt = 0.1
        self.target = np.array([240, 605, 0, 0])

        # generate code
        self.solver = forcespro.nlp.Solver.from_directory("./mpc/MPC_SOLVER/")
        self.N = 25
        self.trajectory = Trajectory2D()
        self.future_trajectory = Trajectory2D()

    def plan(self, drone: Drone2D,
                   dt   : float):
        if len(self.trajectory) != 0:
            return True
        

        x = np.arange(int(self.params.map_size[0]//self.params.map_scale)).reshape(-1, 1) * self.params.map_scale
        y = np.arange(int(self.params.map_size[1]//self.params.map_scale)).reshape(1, -1) * self.params.map_scale
        local_obstacle = np.where(np.logical_and((drone.x - x)**2 + (drone.y - y)**2 <= 100 ** 2, drone.map.grid_map == grid_type['OCCUPIED']), 1, 0)
        positions, widths, heights, angles = self.binary_image_clustering(self, local_obstacle, 1.5, 1, np.array([drone.x, drone.y]))
        

        obs_list = []

        for tracker in drone.trackers:
            if tracker.active:
                obs_list.append([*(tracker.mu_upds[-1][:2,0]), self.params.agent_radius+10, self.params.agent_radius+10, 0, *(tracker.mu_upds[-1][2:,0])])
        
        for i in range(len(positions)):
            obs_list.append([*positions[i], widths[i], heights[i], radians(angles[i]), 0, 0])


        


        problem = {}
        problem["xinit"] = np.array([drone.x, drone.y, *drone.velocity])

        all_params = []

        for i in range(self.N):
            obs_list_i = []
            for obs in obs_list:
                obs_i = obs
                obs_i[0] = obs_i[0] + obs_i[5] * dt * i
                obs_i[1] = obs_i[1] + obs_i[6] * dt * i
                obs_list_i.append(obs_i)

            obs_list_i.sort(key=lambda x: (x[0] - drone.x)**2 + (x[1] - drone.y)**2)

            for j in range(5):
                if j < len(obs_list_i):
                    all_params.extend(obs_list_i[j][:5])
                else:
                    all_params.extend([0]*5)
            all_params.append(self.params.drone_max_speed)
            all_params.append(self.target[0])
            all_params.append(self.target[1])


        problem["all_parameters"] = np.array(all_params)
        # call the solver
        solverout, exitflag, info = self.solver.solve(problem)

        if exitflag != 1:
            self.trajectory = Trajectory2D()
            self.future_trajectory = Trajectory2D()
            return False
        
        self.trajectory = Trajectory2D()
        self.future_trajectory = Trajectory2D()

        # Simulate the system to get the state trajectory
        self.trajectory.positions.append(np.array([solverout["x02"][3], solverout["x02"][4]]))
        self.trajectory.velocities.append(np.array([solverout["x02"][5], solverout["x02"][6]]))
        self.trajectory.accelerations.append(np.array([0, 0]))

        for z in solverout.values():
            self.future_trajectory.positions.append(np.array([z[3], z[4]]))
            self.future_trajectory.velocities.append(np.array([z[5], z[6]]))
            self.future_trajectory.accelerations.append(np.array([0, 0]))

        return True

    @staticmethod
    def binary_image_clustering(self, image, eps, min_samples, start_pos):
        image[0,:] = 0
        image[-1,:] = 0
        image[:,0] = 0
        image[:,-1] = 0
        binary_indices = np.array(np.where(image == 1)).T

        if binary_indices.shape[0] == 0:
            return [],[],[],[]

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(binary_indices)
        labels = dbscan.labels_
        unique_labels = set(labels)

        rs = []
        positions = []
        widths = []
        heights = []
        angles = []

        for k in unique_labels:
            class_member_mask = (labels == k)
            xy = binary_indices[class_member_mask]

            if xy.shape[0] == 1:
                width = 10
                height = 10
                angle = 0
                position = 5 + 10*xy[0]
            else:
                cov = np.cov(xy, rowvar=False)
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
                width, height = 25 * np.sqrt(eig_vals)
                position = 5+10*xy.mean(axis=0)

            ell = Ellipse(position, width, height, angle)
            plt.gca().add_artist(ell)

            widths.append(width)
            heights.append(height)
            positions.append(position)
            angles.append(angle)
            # ell = Circle(position, r)
            # ell = Ellipse(xy.mean(axis=0), r, r, angle, color=col)
        #     plt.gca().add_artist(ell)
        
        # plt.scatter(start_pos[0], start_pos[1], c='r')
        # plt.axis([0,self.params.map_size[0],self.params.map_size[1],0])
        # plt.show()
        # plt.pause(0.1)
        # plt.clf()

        return positions, widths, heights, angles

    def get_coeff(self, p,r_drone,p_obs,r_obs):


        px=p_obs+(r_drone+r_obs)*(p-p_obs)/np.linalg.norm(p-p_obs)

        A = np.array(p - px)
        b = A.dot(px)

        return A,b

    def replan_check(self, drone):
        occupancy_map = drone.map.grid_map
        swep_map = np.zeros_like(occupancy_map)
        for i, pos in enumerate(self.future_trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.params.dt
            for tracker in drone.trackers:
                if tracker.active:
                    if norm(tracker.estimate_pos(i * self.params.dt) - pos) <= self.params.drone_radius + self.params.agent_radius:
                        self.trajectory.clear()
                        return True, swep_map
        if np.sum(np.where((occupancy_map==1),1, 0) * swep_map) > 0:
            self.trajectory.clear()
            return True, swep_map
        return False, swep_map
    
class Jerk_Primitive(Planner):
    def __init__(self, drone, params):
        self.params = params
        self.target = np.zeros(4)
        self.theta_range = np.arange(0, 360, 5)
        self.d = 30
        self.theta_last = - drone.yaw
        self.trajectory = Trajectory2D()
        self.k1 = 1
        self.k2 = 1

    def generate_primitive(self, p0, v0, a0, theta_h, v_max, delt_t):
        delt_x = self.d * np.cos(radians(theta_h))
        delt_y = self.d * np.sin(radians(theta_h))
        pf = p0 + np.array([delt_x, delt_y])

        l = self.target[:2] - pf
        vf = (0.5 * v_max / np.linalg.norm(l)) * l
        af = np.array([0, 0])

        # Choose the time as running in average velocity
        decay_parameter = 0.5
        T1 = delt_x / (vf[0] + v0[0]) 
        T2 = delt_y / (vf[1] + v0[1]) 

        T1 = T1 if T1 < 1000 else 0
        T2 = T2 if T2 < 1000 else 0
        
        T = 1.2 * norm(np.array([delt_x, delt_y])) / (norm(v_max))


        T = T if T >= 0.5 else 0.5

        times = int(np.floor(T / delt_t))
        p = np.zeros((times, 2))
        v = np.zeros((times, 2))
        a = np.zeros((times, 2))
        t = np.arange(delt_t, times * delt_t + delt_t, delt_t)

        # calculate optimal jerk controls by Mark W. Miller
        for ii in range(2):  # x, y axis
            delt_a = af[ii] - a0[ii]
            delt_v = vf[ii] - v0[ii] - a0[ii] * T
            delt_p = pf[ii] - p0[ii] - v0[ii] * T - 0.5 * a0[ii] * T ** 2
            # if vf is not free
            alpha = delt_a * 60 / T ** 3 - delt_v * 360 / T ** 4 + delt_p * 720 / T ** 5
            beta = -delt_a * 24 / T ** 2 + delt_v * 168 / T ** 3 - delt_p * 360 / T ** 4
            gamma = delt_a * 3 / T - delt_v * 24 / T ** 2 + delt_p * 60 / T ** 3

            # if vf is free
            # alpha = -delt_a * 7.5 / T ** 3 + delt_p * 45 / T ** 5
            # beta = delt_a * 7.5 / T ** 2 - delt_p * 45 / T ** 4
            # gamma = -delt_a * 1.5 / T + delt_p * 15 / T ** 3
            for jj in range(times):
                tt = t[jj]
                p[jj, ii] = alpha/120*tt**5 + beta/24*tt**4 + gamma/6*tt**3 + a0[ii]/2*tt**2 + v0[ii]*tt + p0[ii]
                v[jj, ii] = alpha/24*tt**4 + beta/6*tt**3 + gamma/2*tt**2 + a0[ii]*tt + v0[ii]
                a[jj, ii] = alpha/6*tt**3 + beta/2*tt**2 + gamma*tt + a0[ii]
        return p, v, a, t, pf, vf ,af

    def plan(self, drone: Drone2D,
                   dt   : float):
        
        start_pos = np.array([drone.x, drone.y])
        start_vel = drone.velocity
        start_acc = drone.acceleration
        occupancy_map = drone.map
        
        # calculate horizontal offset angle
        delt_p = self.target[:2] - start_pos
        phi_h = math.degrees(atan2(delt_p[1], delt_p[0])) 

        # calculate cost for sampled points
        cost = np.zeros((self.theta_range.shape[0], 2))
        for i, theta in enumerate(self.theta_range):
            cost[i, 0] = self.k1 * (abs(theta % 360 - phi_h % 360) if abs(theta % 360 - phi_h % 360) <= 180 else 360 - abs(theta % 360 - phi_h % 360)) **2
            cost[i, 1] = theta

        # Rank by cost
        cost = cost[cost[:, 0].argsort()]
        v_max = self.params.drone_max_speed

        for seq in range(self.theta_range.shape[0]):
            ps, vs, accs, ts, pf, vf, af = self.generate_primitive(start_pos, start_vel, start_acc, cost[seq, 1], v_max, dt)
            collision = 0
            for t, position in zip(ts, ps):
                if not self.is_free(position, t, occupancy_map, drone.trackers):
                    collision = 1
                    break
            if collision == 0:
                break
        
        if collision:
            return False

        self.trajectory.velocities.append(vs[0, :])
        self.trajectory.accelerations.append(accs[0, :])
        self.trajectory.positions.append(ps[0,:])
        return True

    def replan_check(self, drone):
        occupancy_map = drone.map.grid_map
        swep_map = np.zeros_like(occupancy_map)
        for i, pos in enumerate(self.trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.params.dt
            for tracker in drone.trackers:
                if tracker.active:
                    if norm(tracker.estimate_pos(i * self.params.dt) - pos) <= self.params.drone_radius + self.params.agent_radius:
                        self.trajectory.clear()
                        return True, swep_map
        if np.sum(np.where((occupancy_map==1),1, 0) * swep_map) > 0:
            self.trajectory.clear()
            return True, swep_map
        return False, swep_map
