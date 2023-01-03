import numpy as np
import torch as th
from torch import nn

class CnnEncoder(nn.Module):
    def __init__(
        self, n_input_channels: int, n_output_features: int, sample_input: np.ndarray
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(sample_input[None]).float()).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flatten, n_output_features), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.fc(self.cnn(observations))


class CnnMapEncoder(nn.Module):
    def __init__(
        self, n_input_channels: int, n_output_features: int, sample_input: np.ndarray
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_input_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(sample_input[None]).float()).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flatten, n_output_features), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.fc(self.cnn(observations))