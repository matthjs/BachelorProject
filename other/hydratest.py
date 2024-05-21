from omegaconf import OmegaConf
from hydra import compose, initialize

if __name__ == '__main__':
    # Path to your config file
    config_path = "../util/"

    # Load the config
    with initialize(config_path=config_path):
        cfg = compose(config_name="config")

    # Modify the config as needed
    cfg.model.architecture = "ResNet50"
    cfg.training.batch_size = 64

    # Save the modified config
    OmegaConf.save(cfg, "modified_config.yaml")
