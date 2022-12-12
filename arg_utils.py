import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--gaze-method',
                        default='Oxford',
                        help='Method used (default: Oxford)')
    parser.add_argument('--render',
                        default=True,
                        help='render (default: False)')
    parser.add_argument('--dt',
                        default=0.1,
                        help='environment update time (default: 0.1)')
    parser.add_argument('--map-size',
                        default=(640, 480),
                        help='map size (default: 640*480)')
    parser.add_argument('--map-scale',
                        default=10,
                        help='map compression scale when building gridmap (default: 10 i.e. 64*48 gridmap)')

    parser.add_argument('--agent-number',
                        default=5,
                        help='agent number (default: 5)')
    parser.add_argument('--agent-max-speed',
                        default=20,
                        help='agent max speed (default: 20)')
    parser.add_argument('--agent-radius',
                        default=10,
                        help='agent radius (default: 10)')

    parser.add_argument('--drone-max-speed',
                        default=40,
                        help='drone max speed (default: 40)')
    parser.add_argument('--drone-max-acceleration',
                        default=15,
                        help='drone max acc (default: 15)')
    parser.add_argument('--drone-radius',
                        default=10,
                        help='drone radius (default: 10)')
    parser.add_argument('--drone-max-yaw-speed',
                        default=80,
                        help='drone max yaw speed (default: 80 degrees / s)')
                        
    return parser.parse_args()