import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--gaze-method',
                        default='LookAhead',
                        help='Method used (default: LookAhead)')
    parser.add_argument('--render',
                        default=True,
                        help='render (default: False)')
    parser.add_argument('--dt',
                        default=0.1,
                        help='environment update time (default: 0.1)')
                        
    return parser.parse_args()