#!/usr/bin/env python3

import os
import random
import time
import contextlib
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import env_checker

from smartsim.log import get_logger
import absl.logging

import sys

from smartflow.cfd_env import CFDEnv

import wandb
from wandb.integration.sb3 import WandbCallback

def train(conf, **ignored_kwargs):

    run = wandb.init(
        project="RLWM-Channel",
        name="PPO",
        # config=conf,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    env = CFDEnv(conf)

    if conf.runner.restart:
        model = PPO.load(
            path=f"model_{conf.runner.agent_run_name}",
            env=env,
            custom_objects=None,
        )
    else:
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=3,
            n_steps=conf.runner.n_action_steps_per_pseudo_env_episode,
            batch_size=conf.runner.batch_size,
            tensorboard_log=f"runs/{run.id}"
        )

    # Definition of the learning callback
    checkpoint_callback = CheckpointCallback(
        save_freq=conf.runner.n_action_steps_per_pseudo_env_episode,
        save_path='./logs/',
        name_prefix=f'{conf.logging.run_name}-rl_model'
    )

    # Actual training
    print(f"conf.runner.reset_num_timesteps: {conf.runner.reset_num_timesteps}")
    model.learn(
        total_timesteps=conf.runner.total_pseudo_env_action_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=conf.runner.reset_num_timesteps,
    )

    model.save(
        path=f"model_{conf.runner.agent_run_name}"
    )

    run.finish()