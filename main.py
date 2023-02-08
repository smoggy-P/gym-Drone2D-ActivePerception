import easydict
import time
from experiment import Experiment

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['SDL_AUDIODRIVER'] = 'dsp'

if __name__ == '__main__':

    result_dir = './experiment/results_1.csv'
    img_dir = './experiment/fails/new/'

    cfg = easydict.EasyDict({
        'env':'gym-2d-perception-v2',
        'gaze_method':'Oxford',
        'trained_policy':False,
        'policy_dir':'./trained_policy/lookahead.zip',
        'render':True,
        'dt':0.1,
        'map_scale':10,
        'map_size':[640,480],
        'agent_number':5,
        'agent_max_speed':20,
        'agent_radius':10,
        'drone_max_speed':30,
        'drone_max_acceleration':20,
        'drone_radius':5,
        'drone_max_yaw_speed':80,
        'drone_view_depth' : 80,
        'drone_view_range': 90,
        'record': True,
        'record_img': False,
        'pillar_number':3,
        'img_dir':img_dir,
        'max_steps':8000
    })

    gaze_methods = ['Rotating']
    agent_numbers = [5, 10, 15]
    drone_view_depths = [80]
    drone_view_ranges = [90]
    pillar_numbers = [5, 10]
    agent_max_speeds = [20, 30]
    drone_max_speeds = [20, 30, 40]
    yaw_max_speeds = [80]


    time1 = time.time()
    for gaze_method, agent_number, drone_view_depth, drone_view_range, pillar_number, agent_speed, drone_speed, yaw_speed in zip(gaze_methods, agent_numbers, drone_view_depths, drone_view_ranges, pillar_numbers, agent_max_speeds, drone_max_speeds, yaw_max_speeds):
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
        print("one round finished, time:", time.time()-time1)
        time1 = time.time()