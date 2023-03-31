import easydict
import time
from experiment1 import Experiment
from threading import Thread
import random
import os
from datetime import datetime
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ['SDL_AUDIODRIVER'] = 'dsp'
result_dir = './experiment/results_'+str(datetime.now())+'.csv'
img_dir = './experiment/fails/new/'

def myfun(gaze_method, agent_number, pillar_number, agent_speed, drone_speed, planner, map_id):
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
        'drone_view_range': 360,
        'img_dir':img_dir,
        'max_flight_time': 80,
        'var_cam': 5,


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
    gaze_methods = ['NoControl']
    planners = ['Primitive']
    agent_numbers = [30, 20, 30]
    pillar_numbers = [10]
    agent_max_speeds = [20, 40, 60]
    drone_max_speeds = [40, 40]
    map_ids = range(30)

    
    for gaze_method in gaze_methods:
        for planner in planners:
            for agent_number in agent_numbers:
                for pillar_number in pillar_numbers:
                    for agent_speed in agent_max_speeds:
                        for drone_speed in drone_max_speeds:
                            for map_id in map_ids:
                                myfun(gaze_method, agent_number, pillar_number, agent_speed, drone_speed, planner, map_id)
                        
