import torch
import json

from omegaconf import OmegaConf


class Config:
    # Static fields to store configurations
    GP_FIT_NUM_EPOCHS = 1
    GP_FIT_BATCH_SIZE = 128
    GP_NUM_BATCHES = 50
    GP_FIT_LEARNING_RATE = 0.001
    GP_FIT_RANDOM_BATCHING = True

    UCB_BETA = None
    GP_E_GREEDY_STEPS = None

    DGP_HIDDEN_LAYERS_CONFIG = None

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def initialize_config_object(cfg):
        Config.GP_FIT_NUM_EPOCHS = cfg.fitting.gp_fit_num_epochs
        Config.GP_FIT_BATCH_SIZE = cfg.fitting.gp_fit_batch_size
        Config.GP_NUM_BATCHES = cfg.fitting.gp_fit_num_batches
        Config.GP_FIT_LEARNING_RATE = cfg.fitting.gp_fit_learning_rate
        Config.PG_FIT_RANDOM_BATCHING = cfg.fitting.gp_fit_random_batching

        Config.UCB_BETA = cfg.exploration.ucb_beta
        Config.GP_E_GREEDY_STEPS = cfg.exploration.gp_e_greedy_steps

        print(cfg.dgp_hidden_layers_config)
        Config.DGP_HIDDEN_LAYERS_CONFIG = cfg.dgp_hidden_layers_config
