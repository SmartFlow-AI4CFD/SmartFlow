#!/usr/bin/env python3

from stable_baselines3 import PPO
from smartflow.channel_env import ChannelEnv

def eval(conf, runtime, run, **ignored_kwargs):

    env = ChannelEnv(conf, runtime=runtime)

    model = PPO.load(
        path=conf.runner.model_load_path,
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