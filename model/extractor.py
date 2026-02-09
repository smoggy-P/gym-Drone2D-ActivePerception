"""Custom feature extractors for Stable Baselines 3.

Provides CNN-based observation encoders that process the multi-input
observation space (local map, swept map, yaw angle) produced by the Gym
environment.
"""

from __future__ import annotations

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnEncoder(nn.Module):
    """Small CNN encoder for image-like inputs.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels.
    n_output_features : int
        Dimension of the output feature vector.
    sample_input : np.ndarray
        A sample observation used to infer the flattened size after
        convolutions.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_output_features: int,
        sample_input: np.ndarray,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.as_tensor(sample_input[None]).float()).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, n_output_features), nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(self.cnn(x))


class CnnMapEncoder(nn.Module):
    """Deeper CNN encoder using max-pooling layers.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels.
    n_output_features : int
        Dimension of the output feature vector.
    sample_input : np.ndarray
        A sample observation used to infer the flattened size.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_output_features: int,
        sample_input: np.ndarray,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_input_channels, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.as_tensor(sample_input[None]).float()).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, n_output_features), nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(self.cnn(x))


_CNN_ENCODERS = {
    "CnnEncoder": CnnEncoder,
    "CnnMapEncoder": CnnMapEncoder,
}


class ImgStateExtractor(BaseFeaturesExtractor):
    """Multi-input feature extractor for the drone observation space.

    Processes image observations (local map, swept map) through CNN encoders
    and scalar observations (yaw angle) through a linear layer, then
    concatenates all features.

    Parameters
    ----------
    observation_space : gym.spaces.Dict
        Observation space of the environment.
    device : torch.device
        Computation device.
    cnn_encoder_name : str
        Name of the CNN encoder class (``'CnnEncoder'`` or
        ``'CnnMapEncoder'``).
    cnn_output_dim : int
        Output dimension of each CNN encoder.
    state_output_dim : int
        Output dimension of the scalar state encoder.
    """

    # Keys that should be processed as images
    IMAGE_KEYS = ("swep_map", "local_map")
    # Keys that should be processed as vectors
    VECTOR_KEYS = ("yaw_angle",)
    # Normalisation ranges for vector keys
    VECTOR_SCALES = {"yaw_angle": (0, 360)}

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        device: th.device,
        cnn_encoder_name: str = "CnnEncoder",
        cnn_output_dim: int = 512,
        state_output_dim: int = 32,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        cnn_cls = _CNN_ENCODERS[cnn_encoder_name]
        total_dim = 0
        n_states = 0

        self._img_keys: list[str] = []
        self._vec_keys: list[str] = []
        cnn_list: list[nn.Module] = []

        for key, space in observation_space.spaces.items():
            if key in self.IMAGE_KEYS:
                self._img_keys.append(key)
                cnn_list.append(
                    cnn_cls(space.shape[0], cnn_output_dim, space.sample())
                )
                total_dim += cnn_output_dim
            if key in self.VECTOR_KEYS:
                self._vec_keys.append(key)
                n_states += space.shape[0] if len(space.shape) == 1 else space.shape[1]

        self.cnn_encoders = nn.ModuleList(cnn_list)
        self.state_encoder = nn.Linear(n_states, state_output_dim)
        total_dim += state_output_dim
        self._features_dim = total_dim

    def forward(self, observations: dict) -> th.Tensor:
        parts: list[th.Tensor] = []

        # Image branches
        for i, key in enumerate(self._img_keys):
            img = observations[key].float() / 5.0
            parts.append(self.cnn_encoders[i](img))

        # Scalar branch
        vecs: list[th.Tensor] = []
        for key in self._vec_keys:
            lo, hi = self.VECTOR_SCALES[key]
            normalised = (observations[key] - lo) * 2 / (hi - lo) - 1.0
            vecs.append(normalised)
        if vecs:
            parts.append(self.state_encoder(th.cat(vecs, dim=1)))

        return th.cat(parts, dim=1)
