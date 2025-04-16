#!/usr/bin/env python3

import os
import random
import time
import contextlib
import absl.logging
import sys

import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import env_checker

from smartsim.log import get_logger
from smartflow.channel_env import ChannelEnv


def train(conf, runtime, run, **ignored_kwargs):

    env = ChannelEnv(conf, runtime=runtime)

    if conf.runner.restart:
        model = PPO.load(
            path=conf.runner.model_load_path,
            env=env,
            custom_objects=None,
        )
    else:
        model = PPO(
            policy=conf.runner.policy,
            env=env,
            verbose=3,
            n_steps=conf.runner.steps_per_episode,
            batch_size=conf.runner.batch_size,
            tensorboard_log=f"tensorboard_logs/{run.id}",
            seed=conf.runner.seed,
        )

    # Definition of the learning callback
    checkpoint_callback = CheckpointCallback(
        save_freq=conf.runner.steps_per_episode,
        save_path=os.path.join(conf.runner.model_save_dir, "checkpoints"),
        name_prefix=f"{run.id}",
    )

    # Actual training
    model.learn(
        total_timesteps=conf.runner.total_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=conf.runner.reset_num_timesteps,
    )

    model.save(path=os.path.join(conf.runner.model_save_dir, "final", run.id))

    print("Training finished.")
