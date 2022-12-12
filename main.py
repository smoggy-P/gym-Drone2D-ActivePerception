from arg_utils import get_args
from experiment import Experiment

if __name__ == '__main__':
    cfg = get_args()
    runner = Experiment(cfg)
    runner.run()