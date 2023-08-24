from utils.game import Game
import argparse
import jax
import json
import wandb
import os
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", False)
os.environ["WANDB_MODE"] = "offline"
parser = argparse.ArgumentParser()
parser.add_argument("--param_file")
parser.add_argument("--eta")
parser.add_argument("--step_size")
parser.add_argument("--agent_key")

def main(info):
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

