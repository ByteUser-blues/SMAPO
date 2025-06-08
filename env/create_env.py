import gym
import numpy as np
from pogema import pogema_v0
from pogema.integrations.sample_factory import AutoResetWrapper

from env.wrappers import MultiMapWrapper, ConcatPositionalFeatures, ProjectionTargetWrapper
from learning.learning_config import Environment
from env.SMAPO import SMAPO_preprocessor

class ProvideGlobalObstacles(gym.Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()

def create_env_base(env_cfg: Environment):
    env = pogema_v0(grid_config=env_cfg.grid_config)
    env = ProvideGlobalObstacles(env)
    if env_cfg.use_maps:
        env = MultiMapWrapper(env)  
    return env


def create_env(env_cfg: Environment, auto_reset=False):
    env = create_env_base(env_cfg)
    env = SMAPO_preprocessor(env, env_cfg, auto_reset)
    return env


class MultiEnv(gym.Wrapper):
    def __init__(self, env_cfg: Environment):
        if env_cfg.target_num_agents is None:
            self.envs = [create_env(env_cfg, auto_reset=True)]
        else:
            assert env_cfg.target_num_agents % env_cfg.grid_config.num_agents == 0, "Target num agents must be divisible by num agents"
            num_envs = env_cfg.target_num_agents // env_cfg.grid_config.num_agents
            self.envs = [create_env(env_cfg, auto_reset=True) for _ in range(num_envs)]

        super().__init__(self.envs[0])

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        last_agents = 0   
        for env in self.envs:
            env_num_agents = env.get_num_agents()
            action = actions[last_agents: last_agents + env_num_agents]  
            last_agents = last_agents + env_num_agents
            o, r, d, i = env.step(action)
            obs += o
            rewards += r
            dones += d
            infos += i
        return obs, rewards, dones, infos

    def reset(self):
        obs = []
        for env in self.envs:
            obs += env.reset()
        return obs

    def sample_actions(self):
        actions = []
        for env in self.envs:
            actions += list(env.sample_actions())
        return np.array(actions)

    @property
    def num_agents(self):
        return sum([env.get_num_agents() for env in self.envs])
