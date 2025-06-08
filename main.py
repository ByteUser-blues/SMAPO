from sys import argv
import os
from learning.learning_config import Experiment
from training_run import run
from utils.files import select_free_dir_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def recursive_update(experiment: dict, key, value):
    if key in experiment:
        experiment[key] = value
        return True
    else:
        for k, v in experiment.items():
            if isinstance(v, dict):
                if recursive_update(v, key, value):
                    return True
        return False


def update_dict(target_dict, keys, values):
    for key, value in zip(keys, values):
        if recursive_update(target_dict, key, value):
            print(f'Updated {key} to {value}')
        else:
            print(f'Could not find {key} in experiment')


list_or_args = list(argv)
experiment = Experiment()

if experiment.global_settings.experiments_root is None:
    experiment.global_settings.experiments_root = select_free_dir_name(experiment.global_settings.train_dir)  

experiment = experiment.dict()   
keys = []
values = []
for arg in list_or_args[1:]:
    key, value = arg.split('=')
    key = key.replace('--', '')
    keys.append(key)
    values.append(value)

update_dict(experiment, keys, values)
env_name = experiment['environment']['env']
del experiment['environment']
experiment['environment'] = {'env': env_name}
experiment = Experiment(**experiment).dict()

print('updating keys for environment')
update_dict(experiment['environment'], keys, values)
experiment = Experiment(**experiment)   

if experiment.async_ppo.rollout % experiment.async_ppo.recurrence != 0:
    experiment.async_ppo.rollout = experiment.async_ppo.recurrence
    print(f'Rollout was not divisible by recurrence, setting rollout to {experiment.async_ppo.rollout}')
run(config=experiment.dict(exclude_unset=True))