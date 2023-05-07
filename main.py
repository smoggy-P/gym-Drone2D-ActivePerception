import easydict
import time
from experiment import Experiment
from threading import Thread
import random
import os
from datetime import datetime
from itertools import product
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ['SDL_AUDIODRIVER'] = 'dsp'
result_dir = './experiment/results_'+str(datetime.now())+'.csv'
img_dir = './experiment/fails/new/'

def myfun(gaze_method, agent_number, pillar_number, agent_speed, drone_speed, planner, map_id, motion_profile, var_depth):
    cfg = easydict.EasyDict({
        'env':'gym-2d-perception-v2',
        'render':True,
        'record': False,

        'record_img': False,
        'trained_policy':False,
        'policy_dir':'./trained_policy/lookahead.zip',
        'dt':0.1,
        'map_scale':10,
        'map_size':[480,640],
        'agent_radius':10,
        'drone_max_acceleration':40,
        'drone_radius':5,
        'drone_max_yaw_speed':80,
        'drone_view_depth' : 80,
        'drone_view_range': 90,
        'img_dir':img_dir,
        'max_flight_time': 80,
        
        
        'var_cam': var_depth,
        'motion_profile':motion_profile,
        'gaze_method':gaze_method,#5
        'planner':planner,#3
        'pillar_number':pillar_number,#3
        'agent_number':agent_number,#3
        'drone_max_speed':drone_speed,#3
        'agent_max_speed':agent_speed,#3
        'map_id':map_id
    })
    runner = Experiment(cfg, result_dir)
    runner.run()


if __name__ == '__main__':

    gaze_methods = ['LookAhead', 'Owl', 'Rotating']
    planners = ['Primitive', 'Jerk_Primitive']

    # Environment difficulty
    motion_profiles = ['CVM']
    agent_numbers = [10, 20, 30]
    pillar_numbers = [10]
    agent_max_speeds = [20, 40, 60]
    map_ids = range(5)

    # Problem difficulty
    drone_max_speeds = [20, 40, 60]
    var_depths = [0, 2]
    

    params = product(gaze_methods, planners, motion_profiles, var_depths,
                    agent_numbers, pillar_numbers, agent_max_speeds,
                    drone_max_speeds, map_ids)

    for (gaze_method, planner, motion_profile, var_depth,
        agent_number, pillar_number, agent_speed,
        drone_speed, map_id) in params:
        myfun(gaze_method, agent_number, pillar_number, agent_speed, drone_speed,
            planner, map_id, motion_profile, var_depth)

                        
