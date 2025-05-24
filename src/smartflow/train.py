#!/usr/bin/env python3

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from smartflow.channel_env import ChannelEnv

def train(conf, runtime, run, **ignored_kwargs):

    # Read restart step from the loaded model
    conf.runner.restart_step = 0
    if conf.runner.restart:
        model = PPO.load(
            path=conf.runner.model_load_path,
            custom_objects=None,
        )
        conf.runner.restart_step = model.num_timesteps

    # Start training
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
            learning_rate=conf.runner.learning_rate,
            verbose=2,
            n_steps=conf.runner.steps_per_episode,
            batch_size=conf.runner.batch_size,
            tensorboard_log=f"tensorboard_logs/{run.id}",
            seed=conf.runner.seed,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=conf.runner.save_freq,
        save_path=os.path.join(conf.runner.model_save_dir, "checkpoints"),
        name_prefix=f"{run.id}",
    )

    total_steps = conf.runner.total_steps - conf.runner.restart_step
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=conf.runner.reset_num_timesteps,
    )

    model.save(path=os.path.join(conf.runner.model_save_dir, "final", run.id))

    print("Training finished.")
