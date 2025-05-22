#!/bin/bash

rm -r __pycache__ trajectories envs tensorboard_logs wandb

python -u ../../src/smartflow/main.py \
    runner.model_load_path="/scratch/maochao/code/SmartFlow/experiments/train_retau_05200/models/final/yjkxqlf3" \
    runner.steps_per_episode=3600 \
    > out 2> err
