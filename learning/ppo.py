import json
from os.path import join

from env.replan import RePlanConfig
from learning.encoder_residual import ResnetEncoder
from utils.training_tools import register_custom_components, validate_config

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import numpy as np
from pydantic import Extra

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size, register_custom_encoder
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict
import os
from learning.utils_common import AlgoBase


class PpoConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['SMAPO'] = 'SMAPO'
    path_to_weights: str = "results/SMAPO"
    # here load_weights
    planner_cfg: RePlanConfig = RePlanConfig()

class PpoInference:
    def __init__(self, algo_cfg):
        self.algo_cfg: PpoConfig = algo_cfg

        path = algo_cfg.path_to_weights
        device = algo_cfg.device

        register_custom_components()
        register_custom_encoder('pogema_residual', ResnetEncoder)
        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        self.exp, flat_config = validate_config(config['full_config'])
        algo_cfg = flat_config
        env = create_env(algo_cfg.env, cfg=algo_cfg, env_config={})
        actor_critic = create_actor_critic(algo_cfg, env.observation_space, env.action_space)
        env.close()
        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        actor_critic.model_to_device(device)
        policy_id = algo_cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = algo_cfg

        self.rnn_states = None

    def act(self, observations):
        if self.rnn_states is None:
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)

        with torch.no_grad():

            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)

            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        result = actions.cpu().numpy()
        return result

    def reset_states(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None


def main():
    PpoInference(PpoConfig(path_to_weights='results/mico'))


if __name__ == '__main__':
    main()
