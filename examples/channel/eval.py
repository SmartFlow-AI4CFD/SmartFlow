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

def eval(conf, **ignored_kwargs):

    env = CFDEnv(conf)

    model = PPO.load(
        path=f"model_{conf.runner.agent_run_name}",
        custom_objects=None,
    )

    observations = env.reset()
    for i in range(conf.runner.n_action_steps_per_pseudo_env_episode):
        actions, _states = model.predict(
            observations,
            state=None,
            episode_start=None,
            deterministic=True
        )
        observations, rewards, dones, infos = env.step(actions)