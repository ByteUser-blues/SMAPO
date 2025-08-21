import numpy as np
import gym
import queue

from gym import ObservationWrapper
from gym.spaces import Box, Dict

from env.planning import ResettablePlanner, PlannerConfig
from copy import deepcopy


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01
    add_r = False    
    # when training: False

def SMAPO_preprocessor(env, algo_config, auto_reset):
    env = wrap_preprocessors(env, algo_config.preprocessing, auto_reset)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = SMAPOWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class SMAPOWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy
    
    @staticmethod
    def get_rho(pos_k, pos_oth):
        cur_pos = np.repeat(pos_k[None, :], 64, axis=0)
        rho = pos_oth - cur_pos
        return rho
    
    @staticmethod
    def re_to_ab(re_pos):
        return (re_pos[0] + 5) * 11 + re_pos[1] + 5

    def observation(self, observations):
        
        self.re_plan.update(observations)
        # PP phase:
        paths = self.re_plan.get_path()
        new_goals = []  
        intrinsic_rewards = [] 
        pos_xy = self.grid.positions_xy         
        obstacle = self.grid.obstacles

        ids_oth , relative_xy = self.bfs_obs(pos_xy, obstacle)

        orginal_mask = np.zeros(64)
        pos_xy = np.array(pos_xy)
        # Observations in DR phase:
        for k, path in enumerate(paths):
            obs = observations[k]
            
            if path is None:
                new_goals.append(obs['target_xy'])  
                path = []
            else:
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                # Assign an intrinsic reward if conditions are met, otherwise set it to 0.
                intrinsic_rewards.append(self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0)
                new_goals.append(path[1])
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
    
        for k, _ in enumerate(paths):
            ids_oth_k = np.array(self.Padding(ids_oth[k], 64)).astype(int)
            relative_xy_k = np.array(self.Padding_shape(relative_xy[k], (64, 2))).astype(int)
            observations[k]['ids_oth'] = ids_oth_k
            observations[k]['relative_xy'] =  relative_xy_k
            observations[k]['attention_mask'] = self.create_mask(deepcopy(orginal_mask),len(ids_oth[k]))
            observations[k]['id_'] = k
        
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards
        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
                if self._cfg.add_r == True:
                    reward[agent_idx] += self.intrinsic_reward[agent_idx]
                else:
                    reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.prev_goals = None
        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)   
        self.reset_state()
        return self.observation(observations)
    
    def bfs_obs(self, id_pos, obstacle, d = 5):
        num_agents = len(id_pos)
        dx = [0, 1, -1, 0]
        dy = [1, 0, 0, -1]
        pos_id = dict()
        ids_oth = dict(zip(range(num_agents), [[] for _ in range(num_agents)]))
        relative_xy = dict(zip(range(num_agents), [[] for _ in range(num_agents)]))
        for i , pos in enumerate(id_pos):
            pos_id[pos] = i
        for i in range(num_agents):
            pos = id_pos[i]
            posed = {pos}   
            q = queue.Queue()
            q.put(pos)
            while q.empty() == False:
                x, y = q.get()
                for k in range(4):
                    dx_, dy_ = dx[k], dy[k]
                    nx_x, nx_y = x + dx_, y+dy_
                    tp_nx = (nx_x, nx_y)
                    man_d = self.manhattan_distance(nx_x, nx_y, pos[0], pos[1])
                    if  man_d > d or tp_nx in posed or obstacle[tp_nx] == 1:
                        continue
                    posed.add(tp_nx)
                    q.put(tp_nx)
                    if tp_nx in pos_id.keys():
                        ids_oth[i].append(pos_id[tp_nx] - i)
                        relative_xy[i].append((nx_x - pos[0], nx_y - pos[1]))
        return ids_oth, relative_xy
    
    @staticmethod
    def Padding(x, target_shape):
        length = len(x)
        x = np.array(x)
        if length:
            if length >= 64:
                return x[:64]
            return np.pad(x, (0, target_shape - length))
        else:
            return np.zeros(target_shape)
        
    @staticmethod
    def Padding_shape(x, target_shape):
        length = len(x)
        x = np.array(x)
        if length:
            if length >= 64:
                return x[:64]
            return np.pad(x, pad_width=((0, target_shape[0] - length), (0,0)))
        else:
            return np.zeros(target_shape)
    
    @staticmethod
    def create_mask(orginal_x, mask_length):
        orginal_x[:mask_length] = 1
        return orginal_x
    
    @staticmethod
    def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)


class CutObservationWrapper(ObservationWrapper):
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        return '2_' + x


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, terminated, infos = self.env.step(action)
        if all(terminated):
            observations = self.env.reset()  
        return observations, rewards, terminated, infos
