#!/bin/bash
# shopt -s extglob
# rm -r train __pycache__  experiment DRLsignals
# cd  train-0
# rm -r !(input*|stats*|fld_0.bin)
# cd ..

# cd  train-1
# rm -r !(input*|stats*|fld_0.bin)
# cd ..

# cd  train-2 
# rm -r !(input*|stats*|fld_0.bin)
# cd ..

# python -u main.py > job.out 2> job.err

rm -r __pycache__ trajectories envs tensorboard_logs wandb models/checkpoints

python -u ../../src/smartflow/main.py \
    runner.mode=train \
    runner.restart=False \
    runner.reset_num_timesteps=True \
    > out 2> err
