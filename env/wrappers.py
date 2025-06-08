import gym
import re
from copy import deepcopy
import numpy as np
from numpy import float32
from pogema import GridConfig
from env.custom_maps import MAPS_REGISTRY


class AbstractMetric(gym.Wrapper):
    def _compute_stats(self, step, is_on_goal, truncated):
        raise NotImplementedError

    def __init__(self, env):
        super().__init__(env)
        self._current_step = 0

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        truncated = all(done)
        metric = self._compute_stats(self._current_step, self.was_on_goal, all(done))
        self._current_step += 1
        if truncated:
            self._current_step = 0

        if metric:
            if 'episode_extra_stats' not in infos[0]:
                infos[0]['episode_extra_stats'] = {}
            infos[0]['episode_extra_stats'].update(**metric)

        return obs, reward, done, infos


class MultiMapWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name
        if pattern:
            for map_name in MAPS_REGISTRY:
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)
        cfg = self.grid_config
        if cfg.map_name:
            for agent_idx in range(self.get_num_agents()):
                if 'episode_extra_stats' in info[agent_idx]:
                    for key, value in list(info[agent_idx]['episode_extra_stats'].items()):
                        if key == 'Done':
                            continue
                        info[agent_idx]['episode_extra_stats'][f'{key}-{cfg.map_name.split("-")[0]}'] = value
        return observations, rewards, done, info

    def reset(self, **kwargs):
        if self._configs is not None and len(self._configs) >= 1:
            cfg = deepcopy(self._configs[self._rnd.integers(0, len(self._configs))])
            self.env.unwrapped.grid_config = cfg
        return self.env.reset(**kwargs)


class ProjectionTargetWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space['target_projection'] = deepcopy(self.observation_space['obstacles'])

    @staticmethod
    def get_square_target(x, y, tx, ty, obs_radius):
        full_size = obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        dx, dy = x - tx, y - ty

        dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
        dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
        result[obs_radius - dx, obs_radius - dy] = 1
        return result

    def observation(self, observations):
        obs_radius = observations[0]['obstacles'].shape[0] // 2
        for agent_idx, obs in enumerate(observations):
            target_projection = self.get_square_target(*obs['xy'], *obs['target_xy'], obs_radius)
            observations[agent_idx]['target_projection'] = target_projection

        return observations


class ConcatPositionalFeatures(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        full_size = self.env.observation_space['obstacles'].shape[0]
        self.to_concat = []

        observation_space = gym.spaces.Dict()
        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        observation_space['obs'] = gym.spaces.Box(0.0, 1.0, shape=(len(self.to_concat), full_size, full_size))
        self.to_concat = sorted(self.to_concat, key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=float32)
            observations[agent_idx]['obs'] = main_obs.astype(float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif x == 'agents':
            return '1_' + x
        elif x == 'target_projection':
            return '2_' + x
        return '3_' + x


