#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Optional, List

import time
import numpy as np
from omegaconf import OmegaConf
import random, os

@dataclass
class Runner:
    mode: str
    restart: bool
    reset_num_timesteps: bool = True
    total_env_episodes: int = None
    total_pseudo_env_episodes: int = None  # not applicable for eval mode
    total_pseudo_env_action_steps: int = None  # not applicable for eval mode
    n_epochs: int = 1
    net: tuple = (128, 128)
    learning_rate: float = 5e-4
    log_interval: int = 1  # save model, policy, metrics, interval
    summary_interval: int = 1  # write to tensorboard interval
    seed: Optional[int] = 16
    ckpt_num: int = int(1e6)
    ckpt_interval: int = 1
    n_action_steps_per_pseudo_env_episode: int = None
    n_action_steps_per_training_iteration: int = None
    total_training_iterations: int = None
    agent_run_name: int = 0
    batch_size: int = 1

@dataclass
class Environment:
    n_vec_envs: int
    n_pseudo_envs_per_env: int
    n_pseudo_envs: int = None  # set dynamically after initialization
    n_tasks_per_env: int = 4
    witness_file: str = "witness.txt"
    marl_neighbors: int = 1  # 0 is local state only
    rectangle_file: str = "rectangleControl.txt"
    time_key: str = "time"
    step_type_key: str = "step_type"
    state_key: str = "state"
    state_size_key: str = "state_size"
    action_key: str = "action"
    action_size_key: str = "action_size"
    reward_key: str = "reward"
    poll_time: int = 360000000
    verbosity: str = "debug"  # quiet, debug, info
    dump_data_flag: bool = True
    n_cfd_time_steps_per_action: int = None  # set dynamically after initialization
    agent_interval: int = None  # set dynamically after initialization
    dtype: str = "float32"
    cfd_dtype: str = "float64"
    action_bounds: tuple = (-1.0, 1.0)
    t_action: float = None  # set dynamically after initialization
    f_action: float = None  # calculated from t_action
    t_episode: float = None  # set dynamically after initialization
    t_begin_control: float = 0.0
    reward_norm: int = 1
    reward_beta: float = 0.2  # reward = beta * reward_global + (1.0 - beta) * reward_local
    restart_file: int = 0  # 3: random. 1: restart 1. 2: restart 2
    env_names: List = None
    seed: Optional[int] = 16

@dataclass
class Logging:
    run_name: int = field(default_factory=lambda: int(time.time()))
    group: Optional[str] = None
    notes: Optional[str] = None
    save_dir: str = '../runs'

@dataclass
class SmartSim:
    port: int = field(default_factory=lambda: random.randint(6000, 7000))  # generate a random port number
    n_dbs: int = 1
    network_interface: str = "lo"  # "lo", "ib0"
    run_command: str = "mpirun"
    launcher: str = "local"  # "local", "slurm", "slurm-split"

@dataclass
class Config:
    runner: Runner = None  # needs initialization with required parameters
    environment: Environment = None  # needs initialization with required parameters
    logging: Logging = field(default_factory=Logging)
    smartsim: SmartSim = field(default_factory=SmartSim)

    def __post_init__(self):
        # Initialize with defaults if not provided
        if self.environment is None:
            # These values should be provided by the user or set elsewhere
            n_vec_envs = 3
            n_pseudo_envs_per_env = 48
            n_cfd_time_steps_per_action = 10
            agent_interval = 4
            t_action = 0.1
            t_episode = 0.1
            self.environment = Environment(n_vec_envs=n_vec_envs, n_pseudo_envs_per_env=n_pseudo_envs_per_env)
            self.environment.n_cfd_time_steps_per_action = n_cfd_time_steps_per_action
            self.environment.agent_interval = agent_interval
            self.environment.t_action = t_action
            self.environment.f_action = 1.0 / t_action
            self.environment.t_episode = t_episode
            self.environment.n_pseudo_envs = n_vec_envs * n_pseudo_envs_per_env
            self.environment.env_names = ["retau_1000", "retau_5200"]
        
        if self.runner is None:
            total_env_episodes = 3
            total_pseudo_env_episodes = total_env_episodes * self.environment.n_pseudo_envs_per_env
            n_action_steps_per_pseudo_env_episode = 3
            # These values should be provided by the user or set elsewhere
            self.runner = Runner(mode="train", restart=False)
            self.runner.total_env_episodes = total_env_episodes
            self.runner.n_action_steps_per_pseudo_env_episode = n_action_steps_per_pseudo_env_episode
            self.runner.total_pseudo_env_episodes = total_pseudo_env_episodes
            self.runner.total_pseudo_env_action_steps = total_pseudo_env_episodes * n_action_steps_per_pseudo_env_episode
            self.runner.n_action_steps_per_training_iteration = n_action_steps_per_pseudo_env_episode * self.environment.n_pseudo_envs
            self.runner.total_training_iterations = self.runner.total_pseudo_env_action_steps // self.runner.n_action_steps_per_training_iteration
            self.runner.batch_size = self.runner.n_action_steps_per_training_iteration
            self.runner.reset_num_timesteps = False if self.runner.restart else True
