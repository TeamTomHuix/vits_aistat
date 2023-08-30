from utils.game import Game
import argparse
import jax
import json
import wandb
import os
import hydra
from omegaconf import DictConfig
from gym_examples.envs.run import Trainer

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", False)
os.environ["WANDB_MODE"] = "offline"

@hydra.main(config_path="parameters", config_name="linear")
def main(args: DictConfig):
    wandb.init(config=args)
    game = Game(args)
    game.run_toy()

if __name__ == "__main__":
    main()





