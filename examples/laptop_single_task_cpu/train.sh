#!/bin/bash

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
    environment.cfds_per_case=1 \
    environment.gpus_per_cfd=0 \
    smartsim.network_interface="lo" \
    smartsim.run_command="mpirun" \
    smartsim.launcher="local" \
