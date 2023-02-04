import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CnnEncoder(nn.Module):
    def __init__(
        self, n_input_channels: int, n_output_features: int, sample_input: np.ndarray
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=4, padding=0),
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

class ImgStateExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        device: th.device,
        cnn_encoder_name: str = "CnnEncoder",
        cnn_output_dim: int = 512,
        state_output_dim: int = 32
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(ImgStateExtractor, self).__init__(
            observation_space, features_dim=1
        )

        # Policy Settings

        cnn_class = globals()[cnn_encoder_name]
        self.vector_scales = {
            "yaw_angle": [0, 360],
        }

        # Image input settings
        self.cfg_image_keys = [
            'swep_map',
            'local_map'
        ]

        self.cfg_vector_keys = [
            'yaw_angle'
        ]
        n_states = 0
        total_concat_size = 0

        self.single_img_keys = []
        self.vector_keys = []

        cnn_encoder_single = []

        for key, subspace in observation_space.spaces.items():
            if key in self.cfg_image_keys:
                self.single_img_keys.append(key)
                n_input_channels = subspace.shape[0]
                cnn_encoder_single.append(
                    cnn_class(
                        n_input_channels=n_input_channels,
                        n_output_features=cnn_output_dim,
                        sample_input=subspace.sample(),
                    )
                )
                total_concat_size += cnn_output_dim
            
            if key in self.cfg_vector_keys:
                self.vector_keys.append(key)
                n_states += (
                    subspace.shape[0] if len(subspace.shape) == 1 else subspace.shape[1]
                )

        self.cnn_encoder_single = nn.ModuleList(cnn_encoder_single)

        self.state_encoder = nn.Linear(
            in_features=n_states, out_features=state_output_dim
        )
        total_concat_size += state_output_dim

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # Process Single images
        for i, cnn in enumerate(self.cnn_encoder_single):
            image = observations[self.single_img_keys[i]].float() / 5
            encoded_tensor_list.append(cnn(image))
        
        # Process states
        states_list = []
        for key in self.vector_keys:
            states_list.append(
                (
                    (observations[key] - self.vector_scales[key][0])
                    * 2
                    / (self.vector_scales[key][1] - self.vector_scales[key][0])
                )
                - 1.0
            )
        states = th.cat(states_list, dim=1)
        encoded_tensor_list.append(self.state_encoder(states))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
