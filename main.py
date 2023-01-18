from experiment import Experiment
# from stable_baselines3.common.env_checker import check_env
import easydict

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':

    result_dir = './experiment/results_1.csv'
    img_dir = './experiment/fails/new/'

    cfg = easydict.EasyDict({
        'env':'gym-2d-perception-v1',
        'gaze_method':'Oxford',
        'render':True,
        'dt':0.1,
        'map_scale':10,
        'map_size':[640,480],
        'agent_number':10,
        'agent_max_speed':20,
        'agent_radius':10,
        'drone_max_speed':40,
        'drone_max_acceleration':20,
        'drone_radius':5,
        'drone_max_yaw_speed':80,
        'drone_view_depth' : 80,
        'drone_view_range': 90,
        'record': True,
        'record_img':False,
        'pillar_number':3,
        'img_dir':img_dir
    })

    gaze_methods = ['Oxford']
    agent_numbers = [5, 10]
    drone_view_depths = [80]
    drone_view_ranges = [90]
    pillar_numbers = [10, 15]
    agent_max_speeds = [20, 30]
    drone_max_speeds = [20, 30, 40]



    for gaze_method in gaze_methods:
        for agent_number in agent_numbers:
            for drone_view_depth in drone_view_depths:
                for drone_view_range in drone_view_ranges:
                    for pillar_number in pillar_numbers:
                        for agent_speed in agent_max_speeds:
                            for drone_speed in drone_max_speeds:
                                cfg.gaze_method = gaze_method
                                cfg.agent_number = agent_number
                                cfg.drone_view_depth = drone_view_depth
                                cfg.drone_view_range = drone_view_range
                                cfg.pillar_number = pillar_number
                                cfg.agent_max_speed = agent_speed
                                cfg.drone_max_speed = drone_speed
                                runner = Experiment(cfg, result_dir)
                                runner.run()
    # 
    
    # success, fail = runner.run()

# if __name__ == '__main__':

#     params = get_args()
#     env = gym.make('gym-2d-perception-v0', params=params)  # continuous: LunarLanderContinuous-v2
#     check_env(env)
#     env.is_render = True
#     model = A2C("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10_000)

#     vec_env = model.get_env()
#     vec_env.is_render = True
#     obs = vec_env.reset()

#     for i in range(1000):
#         action, _state = model.predict(obs, deterministic=True)

#         print(action)

#         obs, reward, done, info = vec_env.step(action)
#         vec_env.render()