#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=train
#SBATCH --err=err
#SBATCH --out=out
#SBATCH --account=EUHPC_E04_057
#SBATCH --mail-user=maochao.xiao@uniroma1.it
#SBATCH --mail-type=FAIL
#SBATCH --exclusive

export TMPDIR=/dev/shm

mkdir -p /leonardo_scratch/large/userexternal/mxiao000/wandb/cache
export WANDB_CACHE_DIR="/leonardo_scratch/large/userexternal/mxiao000/wandb/cache"

python -u ~/code/SmartFlow/src/smartflow/main.py \
    wandb.config_file="../config.yaml" \
    runner.mode=train \
    runner.total_cfd_episodes=200 \
    runner.restart=False \
    runner.steps_per_episode=120 \
    environment.cases_dir="../flow_cases" \
    environment.case_names="['retau_05200']" \
    environment.cfd_state_dim=3 \
    environment.agent_state_dim=2 \
    environment.state_definition="log(hwm)+vel" \
    environment.learning_strategy="multi_task" \
    environment.executable_path="/leonardo/home/userexternal/mxiao000/code/SmartFlow/dependencies/CaLES/build/cales" \
    environment.action_start_time=96 \
    environment.time_duration_per_action=0.4 \
    environment.agent_interval=4 \
    environment.agents_per_cfd=768 \
    environment.cfds_per_case=4 \
    smartsim.network_interface="ib0" \
    smartsim.run_command="srun" \
    smartsim.launcher="slurm" \
