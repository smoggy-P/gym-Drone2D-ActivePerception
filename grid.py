import numpy as np
import pygame
from numpy.linalg import norm
from config import *

grid_type = {
    'DYNAMIC_OCCUPIED' : 3,
    'OCCUPIED' : 1,
    'UNOCCUPIED' : 2,
    'UNEXPLORED' : 0
}

class OccupancyGridMap:
    def __init__(self, width, height, dim):
        self.dim = dim
        self.width = width
        self.height = height
        
        self.x_scale = dim[0] / width
        self.y_scale = dim[1] / height
        
        # Define Grid Map
        self.grid_map = np.zeros((self.width, self.height), dtype=np.uint8)

        self.dynamic_idx = []
        
    def init_obstacles(self, obstacles_dict, agents):
        # Mark edges in Grid Map
        self.grid_map[0,:] = grid_type['OCCUPIED']
        self.grid_map[-1,:] = grid_type['OCCUPIED']
        self.grid_map[:,0] = grid_type['OCCUPIED']
        self.grid_map[:,-1] = grid_type['OCCUPIED']
        
        # Mark static obstacles in Grid Map
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                if 'circular_obstacles' in obstacles_dict:
                    for circle in obstacles_dict['circular_obstacles']:
                        if norm(self.get_real_pos(i,j) - np.array([circle[0], circle[1]])) <= circle[2]:
                            self.grid_map[i,j] = grid_type['OCCUPIED']
                if 'rectangle_obstacles' in obstacles_dict:
                    for rect in obstacles_dict['rectangle_obstacles']:
                        if rect[0] <= self.get_real_pos(i,j)[0] <= rect[0] + rect[2] and rect[1] <= self.get_real_pos(i,j)[1] <= rect[1] + rect[3]:
                            self.grid_map[i,j] = grid_type['OCCUPIED']
                if ENABLE_DYNAMIC:
                    for agent in agents:
                        if (self.get_real_pos(i,j)[0] - agent.position[0])**2 + (self.get_real_pos(i,j)[1] - agent.position[1])**2 <= agent.radius ** 2:
                            self.grid_map[i,j] = grid_type['DYNAMIC_OCCUPIED']
                            self.dynamic_idx.append([i,j])
    
    def update_dynamic_grid(self, agents):
        for dynamic_idx in self.dynamic_idx:
            self.grid_map[dynamic_idx[0], dynamic_idx[1]] = grid_type['UNEXPLORED']
        self.dynamic_idx = []
        for agent in agents:
            pos = [int(agent.position[0] // self.x_scale), int(agent.position[1] // self.y_scale)]
            for i in range(max(pos[0] - 1, 0), min(pos[0] + 2, self.grid_map.shape[0])):
                for j in range(max(pos[1] - 1, 0), min(pos[1] + 2, self.grid_map.shape[1])):
                    if self.grid_map[i,j] != grid_type['OCCUPIED']:
                        self.grid_map[i,j] = grid_type['DYNAMIC_OCCUPIED']
                        self.dynamic_idx.append([i,j])

    
    def get_real_pos(self, i, j):
        return np.array([self.x_scale * (i+0.5), self.y_scale * (j+0.5)])
    
    def get_grid(self, x, y):
        return self.grid_map[int(x // self.x_scale), int(y // self.y_scale)]
    
    
    def render(self, surface, color_dict):
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                if(self.grid_map[i,j] == grid_type['OCCUPIED'] or self.grid_map[i,j] == grid_type['DYNAMIC_OCCUPIED']):
                    pygame.draw.rect(surface, color_dict['OCCUPIED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                elif(self.grid_map[i,j] == grid_type['UNOCCUPIED']):
                    pygame.draw.rect(surface, color_dict['UNOCCUPIED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                elif(self.grid_map[i,j] == grid_type['UNEXPLORED']):
                    pygame.draw.rect(surface, color_dict['UNEXPLORED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                
        
    