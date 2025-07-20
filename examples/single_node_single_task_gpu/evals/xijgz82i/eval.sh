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
#SBATCH --array=0-0

export TMPDIR=/dev/shm

mkdir -p /leonardo_scratch/large/userexternal/mxiao000/wandb/cache

export WANDB_CACHE_DIR="/leonardo_scratch/large/userexternal/mxiao000/wandb/cache"

# Define base port for the first task
BASE_PORT=6500
# map each array index to a port and a case name
names=(05200)
idx=$((SLURM_ARRAY_TASK_ID))
port=$((BASE_PORT + idx))
name=${names[$idx]}

EVAL_DIR="eval_retau_${name}"
mkdir -p "$EVAL_DIR" && cd "$EVAL_DIR"

if python -u ~/code/SmartFlow/src/smartflow/main.py \
    wandb.config_file="../../../../config.yaml" \
    runner.mode=eval \
    runner.steps_per_episode=144000 \
    runner.model_load_path="/leonardo/home/userexternal/mxiao000/code/SmartFlow/examples/single_node_single_task_gpu/models/final/xijgz82i.zip" \
    environment.cases_dir="../../../../flow_cases" \
    environment.case_names="['retau_${name}']" \
    environment.cfd_state_dim=3 \
    environment.agent_state_dim=2 \
    environment.state_definition="log(hwm)+vel" \
    environment.learning_strategy="multi_task" \
    environment.executable_path="/leonardo/home/userexternal/mxiao000/code/CaLES/build/cales" \
    environment.action_start_time=0 \
    environment.time_duration_per_action=0 \
    environment.agents_per_cfd=12288 \
    smartsim.network_interface="lo" \
    smartsim.run_command="mpirun" \
    smartsim.launcher="local" \
    smartsim.port=$port \
    extras.hwm_min=0.1 \
    extras.hwm_max=0.1 \
    > out 2> err; then
  cd envs/env_00000
  python ~/code/CaLES/utils/single-point-stats.py
  cd ../..
fi

wait