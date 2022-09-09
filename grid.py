import numpy as np
import pygame

grid_type = {
    'OCCUPIED' : 1,
    'UNOCCUPIED' : 2,
    'UNEXPLORED' : 0
}

class OccupancyGridMap:
    def __init__(self, width, height, dim):
        self.width = width
        self.height = height
        
        self.x_scale = dim[0] / width
        self.y_scale = dim[1] / height
        
        # Define Grid Map
        self.grid_map = np.zeros((self.width, self.height), dtype=np.uint8)
    
    def real_pos(self, i, j):
        return np.array([self.x_scale * i, self.y_scale * j])
    
    def render(self, surface, color_dict):
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                if(self.grid_map[i,j] == grid_type['OCCUPIED']):
                    pygame.draw.rect(surface, color_dict['OCCUPIED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                elif(self.grid_map[i,j] == grid_type['UNOCCUPIED']):
                    pygame.draw.rect(surface, color_dict['UNOCCUPIED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                elif(self.grid_map[i,j] == grid_type['UNEXPLORED']):
                    pygame.draw.rect(surface, color_dict['UNEXPLORED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                
        
    