#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=eval
#SBATCH --err=err_%a
#SBATCH --out=out_%a
#SBATCH --account=EUHPC_E04_057
#SBATCH --mail-user=maochao.xiao@uniroma1.it
#SBATCH --mail-type=FAIL
#SBATCH --array=1-9

export TMPDIR=/dev/shm

mkdir -p /leonardo_scratch/fast/EUHPC_R03_087/wandb/cache

export WANDB_CACHE_DIR="/leonardo_scratch/fast/EUHPC_R03_087/wandb/cache"

EXPERIMENTS_DIR="/leonardo_scratch/fast/EUHPC_R03_087/experiments"

# Define base port for the first task
BASE_PORT=6500
# map each array index to a port and a case name
names=(01000 05200 10000 01E05 01E06 01E07 01E08 01E09 01E10)
idx=$((SLURM_ARRAY_TASK_ID - 1))
port=$((BASE_PORT + idx))
name=${names[$idx]}

EVAL_DIR="eval_retau_${name}"
mkdir -p "$EVAL_DIR" && cd "$EVAL_DIR"

# export CUDA_VISIBLE_DEVICES=$((SLURM_ARRAY_TASK_ID - 1))
# echo "Explicitly setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if python -u ~/code/SmartFlow/src/smartflow/main.py \
    wandb.config_file="../../config.yaml" \
    runner.mode=eval \
    runner.steps_per_episode=3600 \
    runner.model_load_path="$EXPERIMENTS_DIR/train_cpg_medium_mul_4_vel_log_cfl_0.5/models/checkpoints/guqh19b9_103219200_steps.zip" \
    environment.cases_dir="../../flow_cases_cpg_medium_cfl_0.5" \
    environment.case_names="['retau_${name}']" \
    environment.cfd_state_dim=3 \
    environment.agent_state_dim=2 \
    environment.state_definition="log(hwm)+vel" \
    environment.learning_strategy="multi_task" \
    environment.executable_path="/leonardo/home/userexternal/mxiao000/code/CaLES/build/cales" \
    environment.action_start_time=0 \
    environment.time_duration_per_action=0.4 \
    environment.tasks_per_cfd=1 \
    environment.gpus_per_cfd=1 \
    environment.agent_interval=1 \
    environment.agents_per_cfd=3072 \
    smartsim.network_interface="lo" \
    smartsim.run_command="mpirun" \
    smartsim.launcher="local" \
    smartsim.port=$port \
    > out 2> err; then
  cd envs/env_00000
  python ~/code/CaLES/utils/single-point-stats.py
fi

wait
