from argparse import Namespace

import numpy as np

from env.create_env import create_env, MultiEnv
from learning.learning_config import Experiment

from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES, EXTRA_EPISODIC_STATS_PROCESSING
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.utils.utils import log


def make_env(full_env_name, cfg=None, env_config=None):
    if env_config is not None:
        cfg.full_config['environment'].update(**env_config)
    p_config = Experiment(**cfg.full_config)
    environment_config = p_config.environment
    if environment_config.agent_bins is not None and environment_config.target_num_agents is not None:
        if environment_config.env_id is None:
            num_agents = environment_config.agent_bins[0]
        else:
            num_agents = environment_config.agent_bins[environment_config.env_id % len(environment_config.agent_bins)]
        environment_config.grid_config.num_agents = num_agents

        return MultiEnv(environment_config)
    return create_env(environment_config, auto_reset=True)


def validate_config(config):
    exp = Experiment(**config)
    flat_config = Namespace(**exp.async_ppo.dict(),
                            **exp.experiment_settings.dict(),
                            **exp.global_settings.dict(),
                            **exp.evaluation.dict(),
                            **exp.environment.dict(),
                            full_config=exp.dict()
                            )
    return exp, flat_config


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='Pogema',
        make_env_func=make_env,
    )
    EXTRA_EPISODIC_STATS_PROCESSING.append(pogema_extra_episodic_stats_processing)
    EXTRA_PER_POLICY_SUMMARIES.append(pogema_extra_summaries)


def pogema_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    pass


def pogema_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    for key in policy_avg_stats:
        if key in ['reward', 'len', 'true_reward', 'Done']:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f'{policy_id}-{key}: {round(float(avg), 3)}')
