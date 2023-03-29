import easydict
from utils import *
cfg = easydict.EasyDict({
        'env':'gym-2d-perception-v2',
        'render':True,
        'record': False,
        'experiment_time':3,#hours

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
        'var_cam': 5,

        'map_id':0
    })
a = KalmanFilter(params=cfg)
a.mu_upds[0] = 999
b = []
b.append(a.copy())
a.__init__(params=a.params)
print(a.mu_upds[0])
print(b[0].mu_upds[0])
