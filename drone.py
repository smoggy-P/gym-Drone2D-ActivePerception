from grid import OccupancyGridMap
from utils import *
from config import *

class Drone2D():
    def __init__(self, init_x, init_y, init_yaw):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.yaw_range = 120
        self.yaw_depth = 150
        self.radius = DRONE_RADIUS
        self.map = OccupancyGridMap(64, 48, MAP_SIZE)
        
    def render(self, surface):
        pygame.draw.arc(surface, 
                        (255,255,255), 
                        [self.x - self.yaw_depth,
                            self.y - self.yaw_depth,
                            2 * self.yaw_depth,
                            2 * self.yaw_depth], 
                        math.radians(self.yaw - self.yaw_range/2), 
                        math.radians(self.yaw + self.yaw_range/2),
                        2)
        angle1 = math.radians(self.yaw + self.yaw_range/2)
        angle2 = math.radians(self.yaw - self.yaw_range/2)
        pygame.draw.line(surface, (255,255,255), (self.x, self.y), (self.x + self.yaw_depth * cos(angle1), self.y - self.yaw_depth * sin(angle1)), 2)
        pygame.draw.line(surface, (255,255,255), (self.x, self.y), (self.x + self.yaw_depth * cos(angle2), self.y - self.yaw_depth * sin(angle2)), 2)
        pygame.draw.circle(surface, (255,255,255), (self.x, self.y), self.radius)