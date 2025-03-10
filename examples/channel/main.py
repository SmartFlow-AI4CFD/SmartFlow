from omegaconf import OmegaConf
from configuration import Config
from smartflow.train import train
from smartflow.eval import eval

def main():

    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load("config.yaml"),
        OmegaConf.from_cli(),
    )

    print("Configuration:")
    print(OmegaConf.to_yaml(conf))

    if conf.runner.mode == "train":
        train(conf)
    elif conf.runner.mode == "eval":
        eval(conf)

if __name__ == "__main__":
    main()