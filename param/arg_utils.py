import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--gaze-method',
                        default='Oxford',
                        help='Method used (default: Oxford)')
    parser.add_argument('--render',
                        action="store_true",
                        help='render (default: False)')
    parser.add_argument('--dt',
                        type=float,
                        default=0.1,
                        help='environment update time (default: 0.1)')
    parser.add_argument('--map-size',
                        nargs='+',
                        type=int,
                        default=[640, 480],
                        help='map size (default: 640*480)')
    parser.add_argument('--map-scale',
                        default=10,
                        type=int,
                        help='map compression scale when building gridmap (default: 10 i.e. 64*48 gridmap)')

    parser.add_argument('--agent-number',
                        type=int,
                        default=5,
                        help='agent number (default: 5)')
    parser.add_argument('--agent-max-speed',
                        type=int,
                        default=20,
                        help='agent max speed (default: 20)')
    parser.add_argument('--agent-radius',
                        type=int,
                        default=10,
                        help='agent radius (default: 10)')

    parser.add_argument('--drone-max-speed',
                        default=40,
                        type=int,
                        help='drone max speed (default: 40)')
    parser.add_argument('--drone-max-acceleration',
                        default=15,
                        type=int,
                        help='drone max acc (default: 15)')
    parser.add_argument('--drone-radius',
                        default=10,
                        type=int,
                        help='drone radius (default: 10)')
    parser.add_argument('--drone-max-yaw-speed',
                        default=80,
                        type=int,
                        help='drone max yaw speed (default: 80 degrees / s)')
                        
    return parser.parse_args()