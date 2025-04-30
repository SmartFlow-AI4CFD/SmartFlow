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
    poll_time: int = 10000
    save_trajectories: bool = True
    trajectory_dir: str = "trajectories"
    cfd_steps_per_action: int = 10
    agent_interval: int = 4
    cfd_dtype: str = "float64"
    action_bounds: tuple = (-1.0, 1.0)
    reward_beta: float = 0.0
    executable_path: Optional[str] = None
    action_scale_min: float = 0.9
    action_scale_max: float = 1.1
    cfd_state_indices: Optional[List[int]] = None
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
    hidden_layers: tuple = (128, 128)
    learning_rate: float = 3e-4
    seed: Optional[int] = 16
    steps_per_episode: int = 120
    batch_size: Optional[int] = None
    model_load_path: Optional[str] = None
    model_save_dir: str = "models/"
    save_freq: Optional[int] = None
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
    launcher: str = "local"  # "local", "slurm", "slurm-split"


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

    if conf.environment.cfd_state_indices is None:
        conf.environment.cfd_state_indices = list(range(conf.environment.cfd_state_dim))
    
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
        
    return conf
