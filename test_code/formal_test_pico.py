import os
from pogema.animation import AnimationMonitor, AnimationConfig

from learning.ppo import PpoInference, PpoConfig
from env.create_env import create_env
from learning.learning_config import EnvironmentMazes
import torch
import numpy as np
import random
import os
import yaml
import re
import json
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def calculate_mean(numbers):
    total = sum(numbers)
    mean = total / len(numbers)
    return mean

def calculate_standard_deviation(numbers):
    mean = calculate_mean(numbers)
    squared_diffs = [(x - mean) ** 2 for x in numbers]
    variance = sum(squared_diffs) / len(numbers)
    std_deviation = math.sqrt(variance)
    return std_deviation



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def test_on_map(algo, num_agents, map_name):
    
    
    SEED = [0, 1, 2, 3, 42]
    
    res_ = [0 for _ in range(len(SEED))]

    for i in range(len(SEED)):
        seed = SEED[i]
        setup_seed(seed)
        env_cfg = EnvironmentMazes(full_grid=True)
        env_cfg.grid_config.seed = seed
        env_cfg.grid_config.map_name = map_name
        env_cfg.grid_config.num_agents = num_agents
        
        env = create_env(env_cfg, auto_reset=False)
        x = 0
        obs = env.reset()
        algo.reset_states()
        dones = [False, ...]
        while not all(dones):
            action = algo.act(obs)
            obs, rewards, dones, info = env.step(action)
            targets_done = [y >=1 for y in rewards]
            x += sum(targets_done)
        res_[i] = x
    print(res_)
    return res_
    
res = {2**k : {} for k in range(3,9)}
mean = {2**k : {} for k in range(3,9)}
standard_deviation = {2**k : {} for k in range(3,9)}

algo = PpoInference(PpoConfig())
file_path = 'env/random-pico.yaml'
json_path = 'pico.json'

with open(file_path, 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)


# base on old data
if os.path.exists(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    res = json_data[0]
    mean = json_data[1]
    standard_deviation = json_data[2]



num_s = [16, 32, 64, 128]

for num_agent in num_s:
    for i in range(0,21,10):
        name = "pico-" + str(i)
        res[num_agent][name] = []
        pattern = r"^pico_s\w+\_od{}_na32$".format(re.escape(str(i)))
        for name_, _ in zip(data.keys(), data.values()):
            match = re.search(pattern, name_)
            if match:
                print(name_)
                res[num_agent][name].extend(test_on_map(algo, int(num_agent), name_))

        mean[num_agent][name] = calculate_mean(res[num_agent][name])
        standard_deviation[num_agent][name] = calculate_standard_deviation(res[num_agent][name])

load_data = [res, mean, standard_deviation]
with open(json_path, 'w') as file:
    json.dump(load_data, file)

