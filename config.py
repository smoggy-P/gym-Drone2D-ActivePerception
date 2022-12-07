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
MAP_GRID_SCALE = 10

# Dynamic Obstacle Settings
ENABLE_DYNAMIC = True
N_AGENTS = 5
PEDESTRIAN_MAX_SPEED = 20
PEDESTRIAN_RADIUS = 10

# Drone Settings
DRONE_RADIUS = 10
DRONE_MAX_SPEED = 40
DRONE_MAX_ACC = 15
DRONE_MAX_YAW_SPEED = 180

# Ray Casting Settings
