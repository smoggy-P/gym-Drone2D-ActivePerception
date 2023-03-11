
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from math import cos, sin, sqrt, atan2, radians
from sklearn.cluster import DBSCAN
from cvxpy.error import SolverError
from utils import Trajectory2D, Waypoint2D, grid_type

class Planner:
    def __init__(self):
        self.trajectory = Trajectory2D()

    def set_target(self, target):
        self.target = np.zeros(4)
        self.target[:2] = target

    def is_free(self, position, t, occupancy_map, agents):
        if np.isnan(position).any():
            return False
        grid = occupancy_map.get_grid(position[0] - self.params.drone_radius, position[1])
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0], position[1])
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0] + self.params.drone_radius, position[1])
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0], position[1] - self.params.drone_radius)
        if grid == 1:
            return False

        grid = occupancy_map.get_grid(position[0], position[1] + self.params.drone_radius)
        if grid == 1:
            return False

        for agent in agents:
            if agent.seen:
                new_position = agent.estimate_pos + agent.estimate_vel * t
                if norm(position - new_position) <= self.params.drone_radius + agent.radius + 5:
                    return False
        return True

    def plan(self, start_pos, start_vel, start_acc, occupancy_map, agents, update_t):
        raise NotImplementedError("No planner implemented!")

    def replan_check(self, occupancy_map, agents):
        raise NotImplementedError("No replan checker implemented!")

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
        super(Primitive, self).__init__()
        self.params = params
        self.u_space = np.arange(-params.drone_max_acceleration, params.drone_max_acceleration, 0.4 * params.drone_max_speed - 5)
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

    def plan(self, start_pos, start_vel, start_acc, occupancy_map, agents, update_t):
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
        if len(self.trajectory) != 0:
            return True
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
                            if not self.is_free(position, global_t, occupancy_map, agents):
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

    def replan_check(self, occupancy_map, agents):
        swep_map = np.zeros_like(occupancy_map)
        for i, pos in enumerate(self.trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.params.dt
            for agent in agents:
                if agent.seen:
                    if norm(agent.estimated_pos(i * self.params.dt) - pos) <= self.params.drone_radius + agent.radius:
                        self.trajectory.clear()
                        return True, swep_map
        if np.sum(np.where((occupancy_map==1),1, 0) * swep_map) > 0:
            self.trajectory.clear()
            return True, swep_map
        return False, swep_map
    
class MPC(Planner):
    
    def __init__(self, drone, params):
        self.params = params
        self.target = np.array([drone.x, drone.y])
        # Define the prediction horizon and control horizon
        self.N = 15
        self.M = 3

        # Define the state and control constraints
        self.v_max = params.drone_max_speed
        self.x_max = params.map_size[0]
        self.y_max = params.map_size[1]
        self.u_max = params.drone_max_acceleration
        # self.u_max = 80
        self.dt = 0.2

        # Define the system dynamics
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5*self.dt**2, 0],
                           [0, 0.5*self.dt**2],
                           [self.dt, 0],
                           [0, self.dt]])

        # Define the cost function matrices
        self.Q = np.eye(4)
        self.R = np.eye(2)

        self.trajectory = Trajectory2D()
        self.future_trajectory = Trajectory2D()

    @classmethod
    def approx_circle_from_ellipse(self, x0, y0, a, b, theta, x1, y1):

        a = 10 if a <= 0 else a
        b = 10 if b <= 0 else b

        A = inv(np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
        ]))

        import cvxpy as cp

        x = cp.Variable((2))
        constraint = [cp.quad_form(A@(x-np.array([x0, y0])), np.array([[1/a**2,0],[0,1/b**2]])) <= 1]


        cost = cp.norm(x - np.array([x1, y1]))
        prob = cp.Problem(cp.Minimize(cost), constraint)

        # Solve the optimization problem
        result = prob.solve()
        x2, y2 = x.value[0], x.value[1]
        x3, y3 = 2*x0 - x2, 2*y0-y2
        k = -(x1-x2)/(y1-y2)
        r = abs(k*(x3-x2)+y2-y3)/sqrt(k**2+1)/2
        dis = sqrt((x2-x1)**2+(y2-y1)**2)

        x4 = (r/dis)*(x2-x1)+x2
        y4 = (r/dis)*(y2-y1)+y2
        return (x4, y4), r, x2, y2

    @classmethod
    def binary_image_clustering(self, image, eps, min_samples, start_pos):
        image[0,:] = 0
        image[-1,:] = 0
        image[:,0] = 0
        image[:,-1] = 0
        binary_indices = np.array(np.where(image == 1)).T

        if binary_indices.shape[0] == 0:
            return [],[]

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(binary_indices)
        labels = dbscan.labels_
        unique_labels = set(labels)

        rs = []
        positions = []

        for k in unique_labels:
            class_member_mask = (labels == k)
            xy = binary_indices[class_member_mask]

            if xy.shape[0] == 1:
                r = 1.5
                position = 5 + 10*xy[0]
            else:
                cov = np.cov(xy, rowvar=False)
                eig_vals, _ = np.linalg.eigh(cov)
                r =  10 * max(np.sqrt(eig_vals))
                center = 5+10*xy.mean(axis=0)
                position = center
                # ell = Circle(position, r, color='r', fill=False)
                # plt.gca().add_artist(ell)
            rs.append(r)
            positions.append(position)
            ell = plt.Circle(position, r)
            plt.gca().add_artist(ell)
        plt.scatter(start_pos[0], start_pos[1], c='r')
        
        plt.scatter(5+binary_indices[:, 0]*10, 5+binary_indices[:, 1]*10, c='k')
        plt.axis([0,640,480,0])
        
        plt.show()
        plt.pause(0.1)
        plt.clf()

        return positions, rs

    def get_coeff(self, p,r_drone,p_obs,r_obs):


        px=p_obs+(r_drone+r_obs)*(p-p_obs)/np.linalg.norm(p-p_obs)

        A = np.array(p - px)
        b = A.dot(px)

        return A,b

    def plan(self, start_pos, start_vel, start_acc, occupancy_map, agents, dt):
        if len(self.trajectory) != 0:
            return True
        x = np.arange(int(self.params.map_size[0]//self.params.map_scale)).reshape(-1, 1) * self.params.map_scale
        y = np.arange(int(self.params.map_size[1]//self.params.map_scale)).reshape(1, -1) * self.params.map_scale

        local_obstacle = np.where(np.logical_and((start_pos[0] - x)**2 + (start_pos[1] - y)**2 <= 100 ** 2, occupancy_map.grid_map == grid_type['OCCUPIED']), 1, 0)

        positions, rs = self.binary_image_clustering(local_obstacle, 1, 1, start_pos)
        for position, r in zip(positions, rs):
            if norm(start_pos - position) < r:
                return False
        x0 = np.array([start_pos[0], start_pos[1], start_vel[0], start_vel[1]])
        
        # Define the optimization variables
        x = cp.Variable((4, self.N+1))
        u = cp.Variable((2, self.N))
        
        # Define the constraints
        constraints = []
        A_static, b_static = [], []
        for pos, r in zip(positions, rs):
            A_s, b_s = self.get_coeff(x0[:2], self.params.drone_radius, pos, r)
            A_static.append(A_s)
            b_static.append(b_s)

        A_static = np.array(A_static)
        b_static = np.array(b_static)
        
        for i in range(self.N):
            constraints += [x[:,i+1] == self.A@x[:,i] + self.B@u[:,i]]
            if A_static.shape[0] > 0:
                constraints += [A_static@x[:2,i+1] >= b_static.flatten()]

            for agent in agents:
                if agent.seen:
                    p_obs = agent.estimated_pos(i*self.dt)
                    A, b = self.get_coeff(x0[:2], self.params.drone_radius, p_obs, agent.radius + 2)
                    constraints += [A@x[:2,i+1] <= b.flatten()]
            constraints += [15*np.ones(2) <= x[:2,i], x[:2,i] <= np.array(self.params.map_size)-15]
            constraints += [cp.norm(x[2:,i]) <= self.v_max]
            constraints += [cp.norm(u[:,i]) <= self.u_max]
        constraints += [x[:,0] == x0]

        # Define the cost function
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(x[:,i] - self.target, self.Q) + cp.quad_form(u[:,i], self.R)

        # Form the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Solve the optimization problem
        try:
            result = prob.solve(solver='ECOS')
        except SolverError:
            return False
        
        u_opt = u.value
        if u_opt is None:
            return False
        # Simulate the system to get the state trajectory
        self.trajectory = Trajectory2D()

        x = x0
        for i in range(self.N):
            for j in np.arange(0, self.dt, self.params.dt): 
                t = j + self.params.dt
                if i <= self.M:
                    self.trajectory.positions.append(x[:2]+x[2:]*t+0.5*t**2*u_opt[:,i])
                    self.trajectory.velocities.append(x[2:]+t*u_opt[:,i])
                    self.trajectory.accelerations.append(np.array([0, 0]))
                self.future_trajectory.positions.append(x[:2]+x[2:]*t+0.5*t**2*u_opt[:,i])
                self.future_trajectory.velocities.append(x[2:]+t*u_opt[:,i])
                self.future_trajectory.accelerations.append(np.array([0, 0]))
            x = self.A@x + self.B@u_opt[:,i]


        return True

    def replan_check(self, occupancy_map, agents):
        swep_map = np.zeros_like(occupancy_map)
        for i, pos in enumerate(self.future_trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.params.dt
            for agent in agents:
                if agent.seen:
                    if norm(agent.estimated_pos(i * self.params.dt) - pos) <= self.params.drone_radius + agent.radius:
                        self.trajectory.clear()
                        self.future_trajectory.clear()
                        return True, swep_map
        if np.sum(np.where((occupancy_map==1),1, 0) * swep_map) > 0:
            self.trajectory.clear()
            self.future_trajectory.clear()
            return True, swep_map
        return False, swep_map
    
class Jerk_Primitive(Planner):
    def __init__(self, drone, params):
        self.params = params
        self.target = np.zeros(4)
        self.theta_range = np.arange(0, 360, 10)
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

    def plan(self, start_pos, start_vel, start_acc, occupancy_map, agents, dt):
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
                if not self.is_free(position, t, occupancy_map, agents):
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
