#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Optional, List

import time
import numpy as np
from omegaconf import OmegaConf
import random, os


@dataclass
class Wandb:
    project: str
    run_name: str
    sync_tensorboard: bool = True


@dataclass
class Environment:
    n_cfds: int
    agents_per_cfd: int
    cfd_state_dim: int
    cfd_action_dim: int
    cfd_reward_dim: int
    agent_state_dim: int
    agent_action_dim: int
    n_total_agents: int = None  # set dynamically after initialization
    tasks_per_cfd: int = 1
    neighbor_count: int = 0  # 0 is local only
    poll_time: int = 360000000
    verbosity: str = "debug"
    save_trajectories: bool = True
    trajectory_path: str = "trajectories"
    cfd_steps_per_action: int = None  # set dynamically after initialization
    agent_interval: int = None  # set dynamically after initialization
    cfd_dtype: str = "float64"
    action_bounds: tuple = (-1.0, 1.0)
    reward_scale: int = 1
    reward_beta: float = 0.2  # reward = beta * reward_global + (1.0 - beta) * reward_local
    case_names: List = None
    executable_path: str = None
    case_folder: str = "cases"


@dataclass
class Runner:
    mode: str
    restart: bool
    policy: str = "MlpPolicy"
    reset_num_timesteps: bool = True
    total_cfd_episodes: int = None
    total_agent_episodes: int = None  # not applicable for eval mode
    total_steps: int = None  # not applicable for eval mode
    n_epochs: int = 1
    hidden_layers: tuple = (128, 128)
    learning_rate: float = 5e-4
    log_interval: int = 1
    summary_interval: int = 1  # write to tensorboard interval
    seed: Optional[int] = 16
    ckpt_num: int = int(1e6)
    ckpt_interval: int = 1
    steps_per_episode: int = None
    steps_per_batch: int = None
    n_iterations: int = None
    previous_run_id: Optional[int] = None  # Used to specify which model to load when restarting
    batch_size: int = 1


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
    runner: Runner = None  # needs initialization with required parameters
    environment: Environment = None  # needs initialization with required parameters
    logging: Logging = field(default_factory=Logging)
    smartsim: SmartSim = field(default_factory=SmartSim)
    extras: Extras = None
    wandb: Wandb = None  # Wandb configuration

    def __post_init__(self):
        # First initialize environment since runner depends on it
        if self.wandb is None:
            self.wandb = Wandb(
                project="channel",
                run_name=f"run_{self.logging.timestamp}"
            )

        cfd_state_dim = 3
        if self.environment is None:
            n_cfds = 2
            agents_per_cfd = 48
            n_total_agents = n_cfds * agents_per_cfd
            self.environment = Environment(
                n_cfds=n_cfds,
                agents_per_cfd=agents_per_cfd,
                tasks_per_cfd=4,
                cfd_state_dim=cfd_state_dim,
                cfd_action_dim=1,
                cfd_reward_dim=3 + 16,
                agent_state_dim=3,
                agent_action_dim=1,
                cfd_steps_per_action=10,
                agent_interval=4,
                n_total_agents=n_total_agents,
                case_names=["retau_1000", "retau_5200"],
                executable_path="/scratch/maochao/code/CaLES/build/cales",
                save_trajectories=True,
            )

        if self.runner is None:
            total_cfd_episodes = 4
            total_agent_episodes = total_cfd_episodes * agents_per_cfd
            steps_per_episode = 3
            total_steps = total_agent_episodes * steps_per_episode
            steps_per_batch = steps_per_episode * n_total_agents
            
            self.runner = Runner(
                mode="train",
                restart=False,
                total_cfd_episodes=total_cfd_episodes,
                steps_per_episode=steps_per_episode,
                total_agent_episodes=total_agent_episodes,
                total_steps=total_steps,
                steps_per_batch=steps_per_batch,
                n_iterations=total_steps // steps_per_batch,
                batch_size=steps_per_batch,
                reset_num_timesteps=False if self.runner and self.runner.restart else True,
            )

        if self.extras is None:
            self.extras = Extras(n_cells=16)
