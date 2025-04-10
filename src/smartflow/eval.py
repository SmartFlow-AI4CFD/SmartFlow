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
from smartflow.channel_env import ChannelEnv
import absl.logging

import sys

from smartflow.cfd_env import CFDEnv

import wandb
from wandb.integration.sb3 import WandbCallback

def eval(conf, runtime, run, **ignored_kwargs):

    env = ChannelEnv(conf, runtime=runtime)

    model = PPO.load(
        path=f"models/final/{conf.runner.previous_run_id}",
        custom_objects=None,
    )

    observations = env.reset()
    for i in range(conf.runner.steps_per_episode):
        actions, _states = model.predict(
            observations,
            state=None,
            episode_start=None,
            deterministic=True
        )
        observations, rewards, dones, infos = env.step(actions)

    print("Evaluation finished.")