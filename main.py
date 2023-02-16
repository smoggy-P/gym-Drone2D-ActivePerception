import easydict
import time
from experiment import Experiment
from threading import Thread

import os
from datetime import datetime
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ['SDL_AUDIODRIVER'] = 'dsp'
result_dir = './experiment/results_'+str(datetime.now())+'.csv'
# result_dir = './experiment/results_3.csv'
img_dir = './experiment/fails/new/'

def myfun(gaze_method, agent_number, drone_view_depth, drone_view_range, pillar_number, agent_speed, drone_speed, yaw_speed):
    cfg = easydict.EasyDict({
        'env':'gym-2d-perception-v2',
        'render':True,
        'record': False,
        'record_img': False,
        'trained_policy':False,
        'planner':'Jerk_Primitive',#3
        'policy_dir':'./trained_policy/lookahead.zip',

        'gaze_method':'Oxford',#5
        'dt':0.1,
        'map_scale':10,
        'map_size':[640,480],
        'agent_number':5,#3
        'agent_max_speed':20,#3
        'agent_radius':10,
        'drone_max_speed':30,#3
        'drone_max_acceleration':20,
        'drone_radius':5,
        'drone_max_yaw_speed':80,
        'drone_view_depth' : 80,#1
        'drone_view_range': 90,#1
        'pillar_number':3,#3
        'img_dir':img_dir,
        'max_steps':8000
    })
    cfg.gaze_method = gaze_method
    cfg.agent_number = agent_number
    cfg.drone_view_depth = drone_view_depth
    cfg.drone_view_range = drone_view_range
    cfg.pillar_number = pillar_number
    cfg.agent_max_speed = agent_speed
    cfg.drone_max_speed = drone_speed
    cfg.drone_max_yaw_speed = yaw_speed
    runner = Experiment(cfg, result_dir)
    runner.run()


if __name__ == '__main__':
    gaze_methods = ['LookAhead']
    agent_numbers = [15, 15]
    drone_view_depths = [80]
    drone_view_ranges = [90]
    pillar_numbers = [0]
    agent_max_speeds = [20, 30]
    drone_max_speeds = [40, 30, 40]
    yaw_max_speeds = [80]


    ths = []
    for gaze_method in gaze_methods:
        for agent_number in agent_numbers:
            for drone_view_depth in drone_view_depths:
                for drone_view_range in drone_view_ranges:
                    for pillar_number in pillar_numbers:
                        for agent_speed in agent_max_speeds:
                            for drone_speed in drone_max_speeds:
                                for yaw_speed in yaw_max_speeds:
                                    myfun(gaze_method, agent_number, drone_view_depth, drone_view_range, pillar_number, agent_speed, drone_speed, yaw_speed)
                                
