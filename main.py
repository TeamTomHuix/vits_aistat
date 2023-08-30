from utils.game import Game
import argparse
import jax
import json
import wandb
import os
import hydra
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", False)
os.environ["WANDB_MODE"] = "offline"
parser = argparse.ArgumentParser()
parser.add_argument("--param_file")
parser.add_argument("--eta")
parser.add_argument("--step_size")
parser.add_argument("--agent_key")


@hydra.main(config_path="conf", config_name="config")
def main(info):
    def get_args(cfg: DictConfig):
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.hydra_base_dir = os.getcwd()
        return cfg
    wandb.init(config=info)
    game = Game(wandb.config)
    if wandb.config['env'] == 'yahoo':
        game.run_yahoo()
    else:
        game.run_toy()

if __name__ == "__main__":
    args = parser.parse_args()
    info = json.load(open(args.param_file)) 
    sweep_id = wandb.sweep(sweep=info, project=info['parameters']['project_name']['value'])
    wandb.agent(sweep_id, function = lambda : main(info))
    main()



import os
from omegaconf import DictConfig, OmegaConf
import torch 
from gym_examples.envs.run import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    def get_args(cfg: DictConfig):
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.hydra_base_dir = os.getcwd()
        return cfg
    args = get_args(cfg)
    trainer = Trainer(args)
    trainer.run()