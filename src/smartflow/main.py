#!/usr/bin/env python3

from omegaconf import OmegaConf
from smartflow.configuration import Config, compute_derived_parameters
from smartflow.train import train
from smartflow.eval import eval
from smartflow.runtime import Runtime
import wandb
from wandb.integration.sb3 import WandbCallback

def main():

    # Load configuration
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load("config.yaml"),
        OmegaConf.from_cli(),
    )

    conf = compute_derived_parameters(conf)

    print("Configuration:")
    print(OmegaConf.to_yaml(conf))

    wandb_config = OmegaConf.to_container(conf, resolve=True)

    # Initialize wandb
    run = wandb.init(
        project=conf.wandb.project,
        name=conf.wandb.run_name,
        config=wandb_config,
        mode=conf.wandb.mode,
        sync_tensorboard=conf.wandb.sync_tensorboard,
        group=conf.wandb.group if hasattr(conf.wandb, 'group') else None,
        tags=conf.wandb.tags if hasattr(conf.wandb, 'tags') else None,
        save_code=conf.wandb.save_code if hasattr(conf.wandb, 'save_code') else True,
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
