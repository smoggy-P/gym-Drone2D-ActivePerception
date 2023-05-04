import gym
import easydict
import main
import numpy as np
import pandas as pd

def env_metrics(index):

    params = easydict.EasyDict({
            'env':'gym-2d-perception-v2',
            'render':False,
            'record': False,

            'record_img': False,
            'trained_policy':False,
            'policy_dir':'./trained_policy/lookahead.zip',
            'dt':0.1,
            'map_scale':10,
            'map_size':[480,640],
            'agent_radius':10,
            'drone_max_acceleration':40,
            'drone_radius':10,
            'drone_max_yaw_speed':80,
            'drone_view_depth' : 80,
            'drone_view_range': 90,
            'img_dir':'./',
            'max_flight_time': 80,
            

            'gaze_method':'NoControl',#5
            'planner':'NoMove',#3

            'var_cam': 0,
            'drone_max_speed':40,#3

            'motion_profile':index['motion_profile'],
            'pillar_number':index['pillar_number'],
            'agent_number':index['agent_number'],
            'agent_max_speed':index['agent_speed'],
            'map_id':index['map_id']
        })

    position_step = 60
    T = 4
    x_range = range(params.map_scale, params.map_size[0] - params.map_scale, position_step)
    y_range = range(params.map_scale, params.map_size[0] - params.map_scale, position_step)
    total_survive = 0
    for x in x_range:
        for y in y_range:
            env = gym.make(params.env, params=params)
            env.reset()
            for t in np.arange(0, T, 0.1):
                env.drone.x = x
                env.drone.y = y
                _, _, done, info = env.step(0)
                if done:
                    break
            total_survive += t

    return total_survive / (len(x_range) * len(y_range))

env_metric = []

for agent_num in [10, 20, 30]:
    for agent_vel in [20, 40, 60]:
        index = {'motion_profile':'CVM',
                'pillar_number':10,
                'agent_number':agent_num,
                'agent_speed':agent_vel,
                'map_id':0}
        survive_time = env_metrics(index)
        print("average suvival time for agent number ", agent_num, " and agent speed ", agent_vel, "is:", survive_time)
        env_metric.append(survive_time)

metric_dict = {"metric":env_metric}
df = pd.DataFrame(metric_dict)
df.to_csv("metrics.csv")