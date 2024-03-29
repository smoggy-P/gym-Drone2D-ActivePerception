from experiment import Experiment
from datetime import datetime
from itertools import product
import tqdm.contrib.itertools as vis
from utils import *
result_dir = 'experiment/results_'+str(datetime.now())+'.csv'

if __name__ == '__main__':

    gaze_methods = ['LookAhead']
    planners = ['Primitive']

    # Environment difficulty
    motion_profiles = ['CVM']
    agent_numbers = [10, 20, 30]
    agent_sizes = [10, 10, 15]
    agent_max_speeds = [20, 40, 60]
    map_ids = range(5)
    pillar_numbers = [0]

    # Problem difficulty
    drone_max_speeds = [40]
    var_depths = [0]
    

    params = product(gaze_methods, planners, motion_profiles, var_depths,
                    agent_numbers, pillar_numbers, agent_max_speeds,
                    drone_max_speeds, agent_sizes, map_ids)

    i = 1


    for (gaze_method, planner, motion_profile, var_depth,
        agent_number, pillar_number, agent_speed,
        drone_speed, agent_size, map_id) in params:
        axis_range = [250, 460]
        for start_pos, target_pos in vis.product(product(axis_range, axis_range), product(axis_range, axis_range)):
            if start_pos != target_pos:
                cfg = Params.from_parser()
                # cfg = Params(debug=True,
                #             gaze_method=gaze_method, 
                #             planner=planner, 
                #             motion_profile=motion_profile,
                #             var_cam=var_depth,
                #             drone_max_speed=drone_speed,
                #             map_id=map_id,
                #             init_pos=start_pos,
                #             target_list=[target_pos],
                #             agent_number=agent_number,
                #             agent_radius=agent_size,
                #             agent_max_speed=agent_speed,
                #             static_map='maps/empty_map.npy')
                
                i += 1
                if i >= 0:
                    experiment = Experiment(cfg, result_dir)
                    experiment.run()
        
                        
