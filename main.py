from experiment import Experiment
import easydict

if __name__ == '__main__':
    cfg = easydict.EasyDict({
        'gaze_method':'Oxford',
        'render':True,
        'dt':0.1,
        'map_scale':10,
        'map_size':[640,480],
        'agent_number':10,
        'agent_max_speed':20,
        'agent_radius':10,
        'drone_max_speed':40,
        'drone_max_acceleration':15,
        'drone_radius':5,
        'drone_max_yaw_speed':80
    })
    runner = Experiment(cfg)
    success, fail = runner.run()

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