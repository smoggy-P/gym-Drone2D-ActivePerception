import time
from experiment import Experiment
import random
import os
from datetime import datetime
from itertools import product
from utils import *
result_dir = './experiment/results_'+str(datetime.now())+'.csv'

if __name__ == '__main__':

    gaze_methods = ['Owl']
    planners = ['Primitive', 'Jerk_Primitive']

    # Environment difficulty
    motion_profiles = ['CVM']
    agent_numbers = [30, 20, 30]
    agent_sizes = [15, 10, 15]
    agent_max_speeds = [20, 40, 60]
    map_ids = range(30)
    pillar_numbers = [0]

    # Problem difficulty
    drone_max_speeds = [20, 40, 60]
    var_depths = [0, 2]
    

    params = product(gaze_methods, planners, motion_profiles, var_depths,
                    agent_numbers, pillar_numbers, agent_max_speeds,
                    drone_max_speeds, agent_sizes, map_ids)

    for (gaze_method, planner, motion_profile, var_depth,
        agent_number, pillar_number, agent_speed,
        drone_speed, agent_size, map_id) in params:
        start_pos = [100, 50]
        target_list = [[100, 100],[200,200]]
        cfg = Params(debug=True,
                     gaze_method=gaze_method, 
                     planner=planner, 
                     motion_profile=motion_profile,
                     var_cam=var_depth,
                     agent_number=agent_number,
                     pillar_number=pillar_number,
                     agent_max_speed=agent_speed,
                     drone_max_speed=drone_speed,
                     agent_radius=agent_size,
                     map_id=map_id,
                     init_pos=start_pos,
                     target_list=target_list)
        experiment = Experiment(cfg, result_dir)
        experiment.run()
        
                        
