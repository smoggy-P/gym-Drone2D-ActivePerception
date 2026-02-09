#!/usr/bin/env python3
"""Run a single experiment from the command line.

Example
-------
::

    python main.py --gaze_method Oxford --planner Primitive \\
                   --agent_number 10 --agent_max_speed 20 \\
                   --drone_max_speed 40 --map_id 1
"""

from datetime import datetime

from drone2d.config import SimConfig
from experiment import Experiment


def main() -> None:
    config = SimConfig.from_args()
    result_dir = f"experiment/results_{datetime.now()}.csv"
    experiment = Experiment(config, result_dir)
    experiment.run()


if __name__ == "__main__":
    main()
