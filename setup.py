"""Package setup for drone2d."""

from setuptools import setup, find_packages

setup(
    name="drone2d",
    version="1.0.0",
    description="2D Drone Active Perception Gym Environment",
    author="smoggy-P",
    url="https://github.com/smoggy-P/gym-Drone2D-ActivePerception",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "gym>=0.21,<0.26",
        "pygame>=2.1",
        "torch>=1.13",
        "matplotlib>=3.4",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "tqdm>=4.64",
    ],
    extras_require={
        "train": ["stable-baselines3>=1.7", "sb3-contrib>=1.7"],
    },
)
