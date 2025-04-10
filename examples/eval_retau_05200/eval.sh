#!/bin/bash

rm -r __pycache__ trajectories envs tensorboard_logs wandb

python -u ../../src/smartflow/main.py \
    runner.previous_run_id="s6qjwiph" \
    runner.steps_per_episode=3600 \
    > out 2> err
