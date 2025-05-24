#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Wandb:
    project: str = "project_name"
    run_name: str = "run_name"
    mode: str = "online"
    sync_tensorboard: bool = True
    group: Optional[str] = None  # For grouping related experiments
    tags: List[str] = field(default_factory=list)  # For filtering experiments
    save_code: bool = True  # Save code for reproducibility
    run_id: Optional[str] = None  # For resuming runs
    config_file: str = "config.yaml"


@dataclass
class Environment:
    cases_dir: str
    case_names: List[str]
    agents_per_cfd: int
    cfd_state_dim: int
    cfd_action_dim: int
    cfd_reward_dim: int
    agent_state_dim: int
    agent_action_dim: int
    cfds_per_case: int = 1
    tasks_per_cfd: int = 1
    gpus_per_cfd: int = 0
    poll_time: int = 200000
    save_trajectories: bool = True
    trajectory_dir: str = "trajectories"
    action_start_step: Optional[int] = 0
    action_start_time: Optional[float] = 0.0
    cfd_steps_per_action: Optional[int] = 10
    time_duration_per_action: Optional[float] = 0.4
    agent_interval: int = 1
    cfd_dtype: str = "float64"
    action_bounds: tuple = (-1.0, 1.0)
    state_definition: str = "default"
    reward_beta: float = 0.0
    reward_definition: str = "default"
    executable_path: Optional[str] = None
    action_scale_min: float = 0.9
    action_scale_max: float = 1.1
    learning_strategy: str = "sequential"
    # Derived parameters
    n_cfds: Optional[int] = None
    n_agents: Optional[int] = None
    cases_per_batch: Optional[int] = None


@dataclass
class Runner:
    mode: str
    restart: bool
    policy: str = "MlpPolicy"
    reset_num_timesteps: bool = True
    total_cfd_episodes: int = 1
    learning_rate: float = 3e-4
    seed: Optional[int] = 16
    steps_per_episode: int = 120
    batch_size: Optional[int] = None
    model_load_path: Optional[str] = None
    model_save_dir: str = "models/"
    save_freq: Optional[int] = None
    restart_step: Optional[int] = 0
    # Derived parameters
    steps_per_batch: Optional[int] = None
    total_agent_episodes: Optional[int] = None
    total_steps: Optional[int] = None
    total_iterations: Optional[int] = None


@dataclass
class SmartSim:
    port: int = 6379
    n_dbs: int = 1
    network_interface: str = "lo"  # "lo", "ib0"
    run_command: str = "mpirun"
    launcher: str = "local"  # "local", "slurm"
    use_explicit_placement: Optional[bool] = None


@dataclass
class Extras:
    n_cells: int
    tauw_min_percent: Optional[float] = None
    tauw_max_percent: Optional[float] = None
    hwm_min: Optional[float] = None
    hwm_max: Optional[float] = None
    kap_log: Optional[float] = 0.41


@dataclass
class Config:
    wandb: Optional[Wandb] = None
    runner: Optional[Runner] = None
    environment: Optional[Environment] = None
    smartsim: SmartSim = field(default_factory=SmartSim)
    extras: Optional[Extras] = None


def calculate_derived_parameters(conf):
    """Calculate parameters derived from base confuration values for an OmegaConf object."""
    # Environment derived parameters
    if conf.environment.learning_strategy == "sequential":
        conf.environment.cases_per_batch = 1
    elif conf.environment.learning_strategy == "multi_task":
        conf.environment.cases_per_batch = len(conf.environment.case_names)
        
    conf.environment.n_cfds = conf.environment.cases_per_batch * conf.environment.cfds_per_case
    conf.environment.n_agents = conf.environment.n_cfds * conf.environment.agents_per_cfd
    
    # Runner derived parameters
    conf.runner.reset_num_timesteps = not conf.runner.restart
    conf.runner.steps_per_batch = conf.environment.n_agents * conf.runner.steps_per_episode
    conf.runner.total_cfd_episodes = (conf.runner.total_cfd_episodes // conf.environment.n_cfds) * conf.environment.n_cfds
    conf.runner.total_agent_episodes = conf.runner.total_cfd_episodes * conf.environment.agents_per_cfd
    conf.runner.total_steps = conf.runner.total_agent_episodes * conf.runner.steps_per_episode
    conf.runner.total_iterations = conf.runner.total_steps // conf.runner.steps_per_batch
    
    if conf.runner.save_freq is None:
        conf.runner.save_freq = conf.runner.steps_per_episode * 10
        
    if conf.runner.batch_size is None:
        conf.runner.batch_size = conf.runner.steps_per_batch
    
    # Set default for SmartSim use_explicit_placement based on runner mode
    if conf.smartsim.use_explicit_placement is None:
        conf.smartsim.use_explicit_placement = (conf.runner.mode == "train")
        
    return conf
