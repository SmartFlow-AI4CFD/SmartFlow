#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --job-name=train
#SBATCH --err=err
#SBATCH --out=out
#SBATCH --account=EUHPC_E04_057
#SBATCH --mail-user=maochao.xiao@uniroma1.it
#SBATCH --mail-type=FAIL
#SBATCH --exclusive

export TMPDIR=/dev/shm

mkdir -p /leonardo_scratch/fast/EUHPC_R03_087/wandb/cache

# export WANDB_DIR="/leonardo_scratch/fast/EUHPC_R03_087/wandb/logs"
export WANDB_CACHE_DIR="/leonardo_scratch/fast/EUHPC_R03_087/wandb/cache"
# export WANDB_CONFIG_DIR="/leonardo_scratch/fast/EUHPC_R03_087/wandb/configs"

rm -r trajectories envs
python -u ~/code/SmartFlow/src/smartflow/main.py \
    wandb.run_name="train_cpg_medium_mul_4_vel_log_cfl_0.5" \
    wandb.config_file="../config.yaml" \
    runner.mode=train \
    runner.total_cfd_episodes=800 \
    runner.restart=False \
    runner.steps_per_episode=120 \
    runner.model_load_path="models/final/e9to270s" \
    environment.cfds_per_case=1 \
    environment.cases_dir="../flow_cases_cpg_medium_cfl_0.5" \
    environment.case_names="['retau_01000', 'retau_01E05', 'retau_01E07', 'retau_01E10']" \
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
    smartsim.launcher="local"
