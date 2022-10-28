grid_type = {
    'DYNAMIC_OCCUPIED' : 3,
    'OCCUPIED' : 1,
    'UNOCCUPIED' : 2,
    'UNEXPLORED' : 0
}

color_dict = {
    'OCCUPIED'   : (150, 150, 150),
    'UNOCCUPIED' : (50, 50, 50),
    'UNEXPLORED' : (0, 0, 0)
}

# Map Settings
MAP_SIZE = (640, 480)

# Dynamic Obstacle Settings
ENABLE_DYNAMIC = True
N_AGENTS = 5
PEDESTRIAN_MAX_SPEED = 10
PEDESTRIAN_RADIUS = 8

# Drone Settings
DRONE_RADIUS = 4
DRONE_MAX_SPEED = 40

# Ray Casting Settings
