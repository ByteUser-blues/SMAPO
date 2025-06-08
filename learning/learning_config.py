from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pogema import GridConfig
from pydantic import Extra, BaseModel
import os
from os.path import join
from env.SMAPO import PreprocessorConfig

class DMAPFConfig(GridConfig):
    integration: Literal['SampleFactory'] = 'SampleFactory'
    collision_system: Literal['priority', 'block_both'] = 'priority'
    observation_type: Literal['POMAPF'] = 'POMAPF'
    auto_reset: Literal[False] = False

    num_agents: int = 128
    obs_radius: int = 5
    max_episode_steps: int = 512
    map_name: str = r'mazes-.+'


class AsyncPPO(BaseModel, extra=Extra.forbid):   
    experiment_summaries_interval: int = 20
    adam_eps: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    gae_lambda: float = 0.95
    rollout: int = 8
    num_workers: int = 5
    recurrence: int = 8
    use_rnn: bool = True
    rnn_type: str = 'gru'
    rnn_num_layers: int = 1
    ppo_clip_ratio:float = 0.2
    ppo_clip_value: float = 1.0
    batch_size: int = 2048
    num_batches_per_iteration: int = 1
    ppo_epochs: int = 1
    num_minibatches_to_accumulate: int = -1
    max_grad_norm: float = 0.0
    consider_r: int = 2
    
    exploration_loss_coeff: float = 0.023
    atten_hidden_size: int = 256
    value_loss_coeff: float = 0.5
    kl_loss_coeff: float = 0.0
    exploration_loss: str = 'entropy'
    num_envs_per_worker: int = 2
    worker_num_splits: int = 2
    num_policies: int = 1
    policy_workers_per_policy: int = 1
    max_policy_lag: int = 100
    traj_buffers_excess_ratio: int = 30
    decorrelate_experience_max_seconds: int = 10
    decorrelate_envs_on_one_worker: bool = True

    with_vtrace: bool = False
    vtrace_rho: float = 1.0
    vtrace_c: float = 1.0
    set_workers_cpu_affinity: bool = True
    force_envs_single_thread: bool = True
    reset_timeout_seconds: int = 120
    default_niceness: int = 0
    train_in_background_thread: bool = True
    learner_main_loop_num_cores: int = 1
    actor_worker_gpus = []

    with_pbt: bool = False
    pbt_optimize_gamma: bool = True
    pbt_mix_policies_in_one_env: bool = True
    pbt_period_env_steps: int = 3e6
    pbt_start_mutation: int = 2e7
    pbt_replace_fraction: float = 0.3
    pbt_mutation_rate: float = 0.15
    pbt_replace_reward_gap: float = 0.05
    pbt_replace_reward_gap_absolute: float = 1e-6
    pbt_optimize_batch_size: bool = False
    pbt_target_objective: str = 'true_reward'

    use_cpc: bool = False
    cpc_forward_steps: int = 4
    cpc_time_subsample: int = 6
    cpc_forward_subsample: int = 2
    benchmark: bool = False
    sampler_only: bool = False
    max_entropy_coeff: float = 0.0


class ExperimentSettings(BaseModel, extra=Extra.forbid):
    evaluation_saves_per_run: int = 10
    save_every_sec: int = 120
    keep_checkpoints: int = 1
    save_milestones_sec: int = -1
    stats_avg: int = 10
    learning_rate: float = 0.000146
    train_for_env_steps: int = 60000000
    train_for_seconds: int = 1e10

    obs_subtract_mean: float = 0.0
    obs_scale: float = 1.0

    G_sum_beta = 0.8
    gamma: float = 0.9756
    reward_scale: float = 1.0
    reward_clip: float = 10.0

    encoder_type: str = 'resnet'
    encoder_custom: str = 'pogema_residual'
    encoder_subtype: str = 'resnet_impala'
    encoder_extra_fc_layers: int = 1
    G_loss_coeff = 1.0
    num_filters: int = 64
    num_res_blocks: int = 8

    hidden_size: int = 512
    Gfun_hidden_size:int = 256
    G_decoder_layer:int  = 1
    G_mlp_encoder_layer:int = 1
    nonlinearity: str = 'relu'
    policy_initialization: str = 'orthogonal'
    policy_init_gain: float = 1.0
    actor_critic_share_weights: bool = True

    use_spectral_norm: bool = False
    adaptive_stddev: bool = True
    initial_stddev: float = 1.0

    lr_schedule: str = 'kl_adaptive_minibatch'
    lr_schedule_kl_threshold: float = 0.008


class GlobalSettings(BaseModel, extra=Extra.forbid):
    algo: str = 'APPO'
    experiment: str = 'exp'
    experiments_root: str = None
    train_dir: str = 'results/train_dir'
    device: str = 'gpu'
    seed: int = None
    cli_args: dict = {}
    use_wandb: bool = True
    with_wandb: Literal[False] = False


class Evaluation(BaseModel, extra=Extra.forbid):
    fps: int = 0
    render_action_repeat: int = None
    no_render: bool = True
    policy_index: int = 0
    record_to: str = join(os.getcwd(), '..', 'recs')
    continuous_actions_sample: bool = True
    env_frameskip: int = None
    eval_config: str = None


class Environment(BaseModel, ):
    grid_config: DMAPFConfig = DMAPFConfig()
    env: Literal['Pogema-v0'] = "Pogema-v0"
    grid_memory_obs_radius: Optional[int] = None
    observation_type: str = 'POMAPF'
    preprocessing: PreprocessorConfig = PreprocessorConfig()
    sub_goal_distance: Optional[int] = None 
    with_animation: bool = False
    worker_index: int = None
    vector_index: int = None
    env_id: int = None
    target_num_agents: Optional[int] = None
    agent_bins: Optional[list] = [64, 128, 256, 256]
    use_maps: bool = True

    full_grid: bool = True


class EnvironmentMazes(Environment):
    env: Literal['PogemaMazes-v0'] = "PogemaMazes-v0"
    use_maps: bool = True
    target_num_agents: Optional[int] = 256
    agent_bins: Optional[list] = [128, 256, 256, 256]   
    grid_config: DMAPFConfig = DMAPFConfig(on_target='restart', max_episode_steps=512,
                                           map_name=r'mazes-.+')  


class Experiment(BaseModel):
    environment: Union[Environment, EnvironmentMazes,] = EnvironmentMazes()
    async_ppo: AsyncPPO = AsyncPPO()
    experiment_settings: ExperimentSettings = ExperimentSettings()
    global_settings: GlobalSettings = GlobalSettings()
    evaluation: Evaluation = Evaluation()