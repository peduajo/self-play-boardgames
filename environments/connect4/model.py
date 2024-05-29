import torch as th
from torch import nn
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import math

import torch


class DistillNetTrain(nn.Module):
    def __init__(self):
        super(DistillNetTrain, self).__init__()
        
        # Definición de las 3 capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Capa de aplanamiento se hará en el forward
        
        # Definición de la capa lineal
        # Después de las convoluciones, el tamaño del feature map será 6x7x64 (alturaxanchoxcanales)
        # Esto se aplanará a 6*7*64
        self.fc1 = nn.Linear(6*7*64, 128) # 128 es arbitrario, puede ajustarse
        self.fc2 = nn.Linear(128, 1) # Salida única para el valor estimado
        
    def forward(self, x):
        # Aplicación de las capas convolucionales con activación ReLU y max pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Aplanamiento de los feature maps para la capa fully connected
        x = x.view(x.size(0), -1) # Aplana todo excepto el batch size
        
        # Aplicación de las capas lineales con activación ReLU para la primera capa
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No se aplica función de activación aquí para permitir cualquier valor de salida
        
        return x


class DistillNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, row_count, column_count):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(5, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.critic = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * row_count * column_count, 1)
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)

        value = self.critic(x)
        return value

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, num_resBlocks: int = 15, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.startBlock = nn.Sequential(
            nn.Conv2d(5, features_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(features_dim) for i in range(num_resBlocks)]
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.startBlock(observations)
        for resBlock in self.backBone:
            x = resBlock(x)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    

class ResNetDist(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        num_hidden: int = 128,
        row_count: int = 6,
        column_count: int = 7
    ):
        super().__init__()

        self.latent_dim_pi = 32 * column_count * row_count
        self.latent_dim_vf = 3 * row_count * column_count

        self.policyHead = nn.Sequential(
                nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten()
            )
        
        self.valueHeadInt = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.valueHeadExt = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x: th.Tensor):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        v_int, v_ext = self.forward_critic(x)
        return self.forward_actor(x), v_int, v_ext
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        pi = self.policyHead(features)
        return pi

    def forward_critic(self, features: th.Tensor):
        v_int = self.valueHeadInt(features)
        v_ext = self.valueHeadInt(features)
        return v_int, v_ext


class ResNet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        num_hidden: int = 128,
        row_count: int = 6,
        column_count: int = 7,
        noise_layers: bool = False
    ):
        super().__init__()

        self.latent_dim_pi = 32 * column_count * row_count
        self.latent_dim_vf = 3 * row_count * column_count

        self.noisy_layers_p = [
            NoisyLinear(32 * row_count * column_count, 256),
            NoisyLinear(256, column_count)
        ]

        if noise_layers:
            self.policyHead = nn.Sequential(
                nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                self.noisy_layers_p[0],
                nn.ReLU(),
                self.noisy_layers_p[1]
            )
        else:
            self.policyHead = nn.Sequential(
                nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten()
            )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten()
        )

    def sample_noise(self):
        for l in self.noisy_layers_p:
            l.sample_noise()

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        return self.forward_actor(x), self.forward_critic(x)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        pi = self.policyHead(features)
        return pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        v = self.valueHead(features)
        return v


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        features_extractor_class: Type[BaseFeaturesExtractor],
        features_extractor_kwargs: Dict[str, Any],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResNet()


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        if not self.training:
            return super(NoisyLinear, self).forward(input)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()