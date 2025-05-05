#!/usr/bin/env python3

from omegaconf import OmegaConf
import argparse
import wandb

from smartflow.configuration import Config, calculate_derived_parameters
from smartflow.train import train
from smartflow.eval import eval
from smartflow.runtime import Runtime

def main():

    # Merge default config with CLI args to extract the path to config.yaml
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.from_cli(),
    )
    config_file = conf.wandb.config_file

    # Load configuration
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(config_file),
        OmegaConf.from_cli(),
    )
    
    conf = calculate_derived_parameters(conf)
    
    print(OmegaConf.to_yaml(conf))

    wandb_config = OmegaConf.to_container(conf, resolve=True)

    run = wandb.init(
        project=conf.wandb.project,
        name=conf.wandb.run_name,
        config=wandb_config,
        mode=conf.wandb.mode,
        resume="allow",
        sync_tensorboard=conf.wandb.sync_tensorboard,
        group=conf.wandb.group if hasattr(conf.wandb, 'group') else None,
        tags=conf.wandb.tags if hasattr(conf.wandb, 'tags') else None,
        save_code=conf.wandb.save_code if hasattr(conf.wandb, 'save_code') else True,
        id=conf.wandb.run_id,
        settings={
            "_service_wait": 300,
        }
    )
    
    # Define metrics organization
    wandb.define_metric("*", step_metric="global_step")
    # wandb.define_metric("episode/*", step_metric="episode")
    # wandb.define_metric("reward/*", summary="mean")

    # Initialize runtime
    with Runtime(
        type_=conf.smartsim.launcher,
        db_port=conf.smartsim.port,
        db_network_interface=conf.smartsim.network_interface,
        ) as runtime:
    
        runtime.info()

        if conf.runner.mode == "train":
            train(conf, runtime, run=run)
        elif conf.runner.mode == "eval":
            eval(conf, runtime, run=run)

    # Finish wandb
    run.finish()

if __name__ == "__main__":
    main()
