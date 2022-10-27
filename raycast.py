import math
from math import pi, radians, tan, ceil, atan
from config import *

def get_positive_angle(angle = None):

        angle = math.copysign((abs(angle) % (math.pi*2)), angle)
        if (angle < 0):
            angle += (math.pi*2)

        return angle

class Raycast:
    #Pre-calculated values
    rad90deg = radians(90)
    rad270deg = radians(270)

    plane_width = None
    plane_height = None
    distance_to_plane = None
    center_x = None
    center_y = None

    strip_width = 10
    rays_number = None
    rays_angle = None

    world_elem_size = None

    view_angle_tan = None

    def __init__(self,plane_size, drone):
        self.FOV = radians(drone.yaw_range)
        self.depth = drone.yaw_depth
        self.initProjectionPlane(plane_size)

    def initProjectionPlane(self, plane_size):
        self.plane_width, self.plane_height = plane_size
        self.center_x = self.plane_width // 2
        self.center_y = self.plane_height // 2

        self.distance_to_plane = self.center_x / tan(self.FOV/2)

        self.rays_number = ceil(self.plane_width / self.strip_width)
        self.rays_angle = self.FOV / self.plane_width

        self.half_rays_number = self.rays_number//2


    def castRays(self, player, truth_grid_map, updated_grid_map):
        rays = []
        player_angle = pi*2 - radians(player.yaw)
        for i in range(self.rays_number):
            ray_screen_pos = (-self.half_rays_number+i) * self.strip_width
            ray_angle = atan(ray_screen_pos / self.distance_to_plane)
            ray = self.castRay(player, player_angle, ray_angle, truth_grid_map, updated_grid_map)
            rays.append(ray)
        return rays


    def castRay(self, player_coords, player_angle, ray_angle, truth_grid_map, updated_grid_map):   
        # x_step_size = truth_grid_map.x_scale - 1
        # y_step_size = truth_grid_map.y_scale - 1
        x_step_size = 1
        y_step_size = 1
        
        ray_angle = player_angle + ray_angle

        dist = -1
        x_hit = -1
        y_hit = -1
        wall_hit = 0

        #Make shure angle between 0 and 2PI
        ray_angle = get_positive_angle(ray_angle)
        #Get directions which ray is faced
        faced_right = (ray_angle < self.rad90deg or ray_angle > self.rad270deg)
        faced_up = (ray_angle > pi)

        #Find Collision
        slope = tan(ray_angle)
        x = player_coords.x
        y = player_coords.y

        if abs(slope) > 1:
            slope = 1 / slope

            y_step = -y_step_size if faced_up else y_step_size
            x_step = y_step * slope

            while (0 < x < truth_grid_map.dim[0] and 0 < y < truth_grid_map.dim[1]):
                i = int(x // truth_grid_map.x_scale)
                j = int(y // truth_grid_map.y_scale)
                wall = truth_grid_map.grid_map[i, j]
                dist = (x - player_coords.x)**2 + (y - player_coords.y)**2
                if wall == 1 or dist >= self.depth**2:
                    x_hit = x
                    y_hit = y
                    wall_hit = wall
                    if wall == grid_type['OCCUPIED']:
                        updated_grid_map.grid_map[i, j] = grid_type['OCCUPIED']
                    break
                else:
                    updated_grid_map.grid_map[i, j] = grid_type['UNOCCUPIED']
                x = x + x_step
                y = y + y_step
        
        else:
            x_step = x_step_size if faced_right else -x_step_size
            y_step = x_step * slope
            
            while (0 < x < truth_grid_map.dim[0] and 0 < y < truth_grid_map.dim[1]):
                i = int(x // truth_grid_map.x_scale)
                j = int(y // truth_grid_map.y_scale)
                wall = truth_grid_map.grid_map[i, j]
                dist = (x-player_coords.x)**2 + (y-player_coords.y)**2
                if wall == 1 or dist >= self.depth**2:
                    x_hit = x
                    y_hit = y
                    wall_hit = wall
                    if wall == grid_type['OCCUPIED']:
                        updated_grid_map.grid_map[i, j] = grid_type['OCCUPIED']
                    break
                else:
                    updated_grid_map.grid_map[i, j] = grid_type['UNOCCUPIED']

                x = x + x_step
                y = y + y_step
        
        result = {'coords':(x_hit,y_hit), 'wall':wall_hit}
        return result