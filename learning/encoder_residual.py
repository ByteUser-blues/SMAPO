import torch
from typing import Literal
from learning.learning_config import ExperimentSettings
from sample_factory.algorithms.appo.model_utils import get_obs_shape, EncoderBase, ResBlock, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

from sample_factory.utils.utils import log
from pydantic import BaseModel
from torch import nn as nn


def activation_func(cfg) -> nn.Module:
    """
    Returns an instance of nn.Module representing the activation function specified in the configuration.

    Args:
        cfg (EncoderConfig): Encoder configuration.

    Returns:
        nn.Module: Instance of nn.Module representing the activation function.

    Raises:
        Exception: If the activation function specified in the configuration is unknown.
    """
    if cfg.nonlinearity == "rlu":
        return nn.ELU(inplace=True)
    elif cfg.nonlinearity == "relu":
        return nn.ReLU(inplace=True)
    elif cfg.nonlinearity == "mish":
        return nn.Mish(inplace=True)
    else:
        raise Exception("Unknown activation_func")

class ResBlock(nn.Module):
    """
    Residual block in the encoder.

    Args:
        cfg (EncoderConfig): Encoder configuration.
        input_ch (int): Input channel size.
        output_ch (int): Output channel size.
    """

    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            activation_func(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
            activation_func(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        # noinspection Pydantic

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]

        resnet_conf = [[cfg.num_filters, cfg.num_res_blocks]]
        curr_input_channels = input_ch
        layers = []

        for out_channels, res_blocks in resnet_conf:
            layers.extend([nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)])
            layers.extend([ResBlock(cfg, out_channels, out_channels) for _ in range(res_blocks)])
            curr_input_channels = out_channels

        layers.append(activation_func(cfg))
        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        self.encoder_out_size = self.conv_head_out_size

        if cfg.encoder_extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, cfg.hidden_size),
                activation_func(cfg),
            )
            self.encoder_out_size = cfg.hidden_size
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        self.coordinates_mlp = nn.Sequential(
            nn.Linear(4, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )

        self.init_fc_blocks(self.conv_head_out_size + cfg.hidden_size)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        x = x['obs']
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)

        if self.cfg.encoder_extra_fc_layers:
            x = self.extra_linear(x)

        return x
