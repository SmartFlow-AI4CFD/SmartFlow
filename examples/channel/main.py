from omegaconf import OmegaConf
from configuration import Config
from train import train
from eval import eval
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

    print("Configuration:")
    print(OmegaConf.to_yaml(conf))

    # Initialize wandb
    run = wandb.init(
        project=conf.wandb.project,
        # id = "PPO-first",
        name=conf.wandb.run_name,
        # config=conf,
        sync_tensorboard=conf.wandb.sync_tensorboard, # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    wandb.define_metric("*", step_metric="global_step")

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
            eval(conf)

    # Finish wandb
    run.finish()

if __name__ == "__main__":
    main()
