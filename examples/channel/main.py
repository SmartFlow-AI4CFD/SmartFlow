from omegaconf import OmegaConf
from configuration import Config
from smartflow.train import train
from smartflow.eval import eval
from smartflow.runtime import Runtime

def main():

    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load("config.yaml"),
        OmegaConf.from_cli(),
    )

    print("Configuration:")
    print(OmegaConf.to_yaml(conf))

    with Runtime(
        type_=conf.smartsim.launcher,
        db_port=conf.smartsim.port,
        db_network_interface=conf.smartsim.network_interface,
        ) as runtime:
    
        runtime.info()

        if conf.runner.mode == "train":
            train(
                conf,
                runtime,
            )
        elif conf.runner.mode == "eval":
            eval(conf)

if __name__ == "__main__":
    main()