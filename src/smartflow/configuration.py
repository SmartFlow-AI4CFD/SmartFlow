#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional, List

import time
import numpy as np
from omegaconf import OmegaConf
import random, os


@dataclass
class Wandb:
    project: str = "project_name"
    run_name: str = "run_name"
    mode: str = "online"
    sync_tensorboard: bool = True
    group: Optional[str] = None  # For grouping related experiments
    tags: List[str] = field(default_factory=list)  # For filtering experiments
    save_code: bool = True  # Save code for reproducibility

@dataclass
class Environment:
    n_cfds: int
    agents_per_cfd: int
    cfd_state_dim: int
    cfd_action_dim: int
    cfd_reward_dim: int
    agent_state_dim: int
    agent_action_dim: int
    n_total_agents: Optional[int] = None  # set dynamically after initialization
    tasks_per_cfd: int = 1
    neighbor_count: int = 0  # 0 is local only
    poll_time: int = 360000000
    verbosity: str = "debug"
    save_trajectories: bool = True
    trajectory_path: str = "trajectories"
    cfd_steps_per_action: Optional[int] = None  # set dynamically after initialization
    agent_interval: Optional[int] = None  # set dynamically after initialization
    cfd_dtype: str = "float64"
    action_bounds: tuple = (-1.0, 1.0)
    reward_scale: int = 1
    reward_beta: float = 0.2  # reward = beta * reward_global + (1.0 - beta) * reward_local
    case_names: Optional[List[str]] = None
    executable_path: Optional[str] = None
    case_folder: str = "flow_cases"


@dataclass
class Runner:
    mode: str
    restart: bool
    policy: str = "MlpPolicy"
    reset_num_timesteps: bool = True
    total_cfd_episodes: Optional[int] = None
    total_agent_episodes: Optional[int] = None
    total_steps: Optional[int] = None
    n_epochs: int = 1
    hidden_layers: tuple = (128, 128)
    learning_rate: float = 5e-4
    log_interval: int = 1
    summary_interval: int = 1  # write to tensorboard interval
    seed: Optional[int] = 16
    ckpt_num: int = int(1e6)
    ckpt_interval: int = 1
    steps_per_episode: Optional[int] = None
    steps_per_batch: Optional[int] = None
    n_iterations: Optional[int] = None
    batch_size: int = 1
    previous_run_id: Optional[str] = None


@dataclass
class Logging:
    timestamp: int = field(default_factory=lambda: int(time.time()))
    save_dir: str = '../runs'


@dataclass
class SmartSim:
    port: int = field(default_factory=lambda: random.randint(6000, 7000))  # generate a random port number
    n_dbs: int = 1
    network_interface: str = "lo"  # "lo", "ib0"
    run_command: str = "mpirun"
    launcher: str = "local"  # "local", "slurm", "slurm-split"


@dataclass
class Extras:
    n_cells: int


@dataclass
class Config:
    runner: Optional[Runner] = None  # needs initialization with required parameters
    environment: Optional[Environment] = None  # needs initialization with required parameters
    logging: Logging = field(default_factory=Logging)
    smartsim: SmartSim = field(default_factory=SmartSim)
    extras: Optional[Extras] = None
    wandb: Optional[Wandb] = None  # Wandb configuration


def compute_derived_parameters(conf):
    """Process the configuration after merging all sources, calculating derived values."""
    # Calculate agent-related values
    if conf.environment is not None:
        # Set n_total_agents if not explicitly set
        if conf.environment.n_total_agents is None:
            conf.environment.n_total_agents = conf.environment.n_cfds * conf.environment.agents_per_cfd
            
        # Ensure all required environment parameters have values
        if conf.environment.cfd_steps_per_action is None:
            conf.environment.cfd_steps_per_action = 10
            
        if conf.environment.agent_interval is None:
            conf.environment.agent_interval = 4
    
    # Fill in missing runner parameters
    if conf.runner is not None and conf.environment is not None:
        # Set derived runner parameters based on environment
        if conf.runner.total_cfd_episodes is not None:
            # Only calculate these for training mode
            if conf.runner.mode == "train" or conf.runner.mode == "eval":
                n_total_agents = conf.environment.n_total_agents
                agents_per_cfd = conf.environment.agents_per_cfd
                
                # Set total_agent_episodes if not explicitly set
                if conf.runner.total_agent_episodes is None:
                    conf.runner.total_agent_episodes = conf.runner.total_cfd_episodes * agents_per_cfd
                
                # Set steps_per_episode if not explicitly set
                if conf.runner.steps_per_episode is None:
                    conf.runner.steps_per_episode = 120
                
                # Set total_steps if not explicitly set
                if conf.runner.total_steps is None:
                    conf.runner.total_steps = conf.runner.total_agent_episodes * conf.runner.steps_per_episode
                
                # Set steps_per_batch if not explicitly set
                if conf.runner.steps_per_batch is None:
                    conf.runner.steps_per_batch = conf.runner.steps_per_episode * n_total_agents
                
                # Set n_iterations if not explicitly set
                if conf.runner.n_iterations is None:
                    conf.runner.n_iterations = conf.runner.total_steps // conf.runner.steps_per_batch
                
                # Override batch_size to match steps_per_batch
                conf.runner.batch_size = conf.runner.steps_per_batch
        
        # Override reset_num_timesteps based on restart flag
        conf.runner.reset_num_timesteps = not conf.runner.restart
    
    return conf
