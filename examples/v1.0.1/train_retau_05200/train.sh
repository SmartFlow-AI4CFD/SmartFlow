rm -r __pycache__ trajectories envs tensorboard_logs wandb models/checkpoints

python -u ../../src/smartflow/main.py \
    runner.mode=train \
    runner.total_cfd_episodes=400 \
    runner.restart=False \
    runner.reset_num_timesteps=True \
    > out 2> err
