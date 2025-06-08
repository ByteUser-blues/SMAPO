import json
import os
from argparse import Namespace
from pathlib import Path

from os.path import join, exists
from shutil import rmtree

import numpy as np
import yaml

# from eval import evaluate
from learning.encoder_residual import ResnetEncoder
from sample_factory.algorithms.appo.model_utils import register_custom_encoder

from sample_factory.utils.utils import log, summaries_dir, experiment_dir

import wandb
from sample_factory.run_algorithm import run_algorithm

from utils.files import select_free_dir_name
from utils.training_tools import validate_config, register_custom_components


def get_summary_metrics(path_to_tensorboard):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(path_to_tensorboard)
    event_acc.Reload()

    keys = [key for key in event_acc.Tags()['scalars'] if '/' not in key]

    results = {}

    for key in keys:
        values = event_acc.Scalars(key)[-len(event_acc.Scalars(key)) // 4:]
        values = [v.value for v in values][-int(len(values) / 4):]
        if len(values) > 0:
            results[f'last_quartile/{key}'] = np.mean(values)
    log.info(f'Last quartile metrics: {results}')

    return results


def evaluate_policy(policy_evaluation_dir):
    pass


def run(config=None):
    register_custom_encoder('pogema_residual', ResnetEncoder)
    params = Namespace(**config)  
    params.wandb_thread_mode = False

    register_custom_components()
    exp, flat_config = validate_config(config)
    log.info(flat_config)

    if exp.global_settings.experiments_root is None:
        exp.global_settings.experiments_root = select_free_dir_name(exp.global_settings.train_dir)
        exp, flat_config = validate_config(exp.dict())
    log.debug(exp.global_settings.experiments_root)
    
    if exp.global_settings.use_wandb:
        import os
        if params.wandb_thread_mode:
            os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project=exp.environment.env, config=exp.dict(), save_code=False, sync_tensorboard=True,
                   anonymous="allow", )
    
    status = run_algorithm(flat_config)
    
    last_quartile_metrics = get_summary_metrics(summaries_dir(experiment_dir(cfg=flat_config)) + '/0', )
    if exp.global_settings.use_wandb:
        import shutil
        path = Path(exp.global_settings.train_dir) / exp.global_settings.experiments_root
        zip_name = str(path)
        shutil.make_archive(zip_name, 'zip', path)
        wandb.save(zip_name + '.zip')

        wandb.log(last_quartile_metrics)
        wandb.finish()

    return status
