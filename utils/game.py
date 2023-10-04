import jax
import jax.numpy as jnp
import wandb
import json
from agents.lints import LinTS
from agents.linucb import LinUCB
from agents.lmcts import LMCTS
from agents.vits import VITS
from agents.vts import VTS
from agents.rvits import RVITS
from agents.random import Random
from utils.utils import UtilsVector
from env import LinearDataset, LogisticDataset, YahooEnv, LinearIllDataset, LogisticIllDataset
from tqdm import tqdm
import pickle as pkl

class Game(object):
    def __init__(self, info):

        self.agent_dict = {
            "ts": LinTS,
            "ucb": LinUCB,
            "vits": VITS,
            "lmcts": LMCTS,
            "vts": VTS,
            "rvits": RVITS,
            "random": Random
        }

        self.env_dict = {
            "linear": LinearDataset,
            "linearill": LinearIllDataset,
            "logistic": LogisticDataset,
            "logisticill": LogisticIllDataset,
            "yahoo": YahooEnv,
        }
        self.info = info
        self.theta_key = jax.random.PRNGKey(self.info.key.theta)
        self.agent_key = jax.random.PRNGKey(self.info.key.agent)
        self.data_key = jax.random.PRNGKey(self.info.key.data)
        self.init_agent()
        self.init_env()

    def run_toy(self):
        cum_regret = jnp.zeros(self.info.T)
        for idx in range(self.info.T):
            context = self.context_fct(idx)
            self.agent_key, utils_vector, action = self.choice_fct(self.agent_key, context, self.utils_object.get())
            self.utils_object.set(utils_vector) 
            self.data_key, reward, expected_reward, best_expected_reward = self.reward_fct(idx, self.data_key, action)
            self.agent_key, utils_vector = self.update_fct(self.agent_key, context, action, reward, self.utils_object.get(idx + 1))
            self.utils_object.set(utils_vector, idx + 1)
            cum_regret = cum_regret.at[idx + 1].set(cum_regret[idx] + best_expected_reward - expected_reward)
            #condition_number = self.compute_cond_number(self.utils_object.get(idx+1))
            wandb.log({"cum_regret": jax.device_get(cum_regret[idx]),
                    "action": jax.device_get(action),
                    "reward": jax.device_get(reward),
                    "is_arm_0": 1 if action == 0 else 0,
                    "is_arm_1": 1 if action == 1 else 0,
                    #"condition_number": jax.device_get(condition_number)
                    })
        wandb.finish()
        return cum_regret

    def run_yahoo(self):
        G_learn, T_learn = 0, 0 
        for idx in range(self.info.T):
            displayed_arm, context = self.context_fct(idx)
            self.agent_key, utils_vector, action = self.choice_fct(self.agent_key, context, self.utils_object.get())
            self.utils_object.set(utils_vector)
            if displayed_arm == action:
                self.data_key, reward, _, _ = self.reward_fct(idx, self.data_key, action)
                G_learn += reward
                T_learn += 1
                self.agent_key, utils_vector = self.update_fct(self.agent_key, context, action, reward, self.utils_object.get(idx + 1))
                self.utils_object.set(utils_vector, idx + 1)
                wandb.log({"ratio CTR": jax.device_get(G_learn / T_learn),
                           "action": jax.device_get(action),
                           "reward": jax.device_get(reward)})
        wandb.finish()
        return G_learn / T_learn

    def init_agent(self):
        self.agent_key, key_utils = jax.random.split(self.agent_key)
        self.utils_object = UtilsVector(self.info, key_utils)
        agent = self.agent_dict[self.info.agent_name](self.utils_object.info, self.utils_object)
        self.choice_fct = jax.jit(agent.choice_fct)
        self.update_fct = jax.jit(agent.update_fct)
        #self.compute_cond_number = jax.jit(agent.compute_cond_number)

    def init_env(self):
        self.data_key, data_key_temp = jax.random.split(self.data_key)
        self.environment = self.env_dict[self.info.env](self.info, self.theta_key, data_key_temp)
        self.context_fct = jax.jit(self.environment.context_fct)
        self.reward_fct = jax.jit(self.environment.reward_fct)
        
