import jax
import jax.numpy as jnp
import pandas as pd
import requests
from io import BytesIO
import numpy as np
from jax.scipy.linalg import block_diag
import fileinput
from os.path import exists
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

class LinearDataset(object):
    def __init__(self, info, theta_key, data_key):
        self.info = info
        self.theta = self.init_theta_star(theta_key)
        self.contexts, self.mean, self.noise = self.generate_data(data_key)
       

    def init_theta_star(self, theta_key):
        theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1)) / np.sqrt(self.info.ctx_dim)
        #theta /= jnp.linalg.norm(theta, ord=2)
        return theta

    def generate_data(self, data_key):
        key, subkey = jax.random.split(data_key)
        pool = jax.random.normal(subkey, shape=(50, self.info.ctx_dim))
        key, subkey = jax.random.split(key)
        indexes = jax.random.randint(subkey, minval=0, maxval=50, shape=(self.info.T, self.info.nb_arms))
        key, subkey = jax.random.split(key)
        contexts = pool[indexes] + self.info.context_noise * jax.random.normal(subkey, shape=(self.info.T, self.info.nb_arms, self.info.ctx_dim))
        mean = (contexts @ self.theta).squeeze()
        key, subkey = jax.random.split(key)
        noise = np.sqrt(1 + self.info.context_noise**2) * jax.random.normal(subkey, shape=(self.info.T,))
        return contexts, mean, noise

    def reward_fct(self, idx, data_key, action):
        reward = self.mean[idx, action] + self.noise[idx]
        expected_reward = self.mean[idx, action].squeeze()
        best_expected_reward = jnp.max(self.mean[idx, :])
        return data_key, reward, expected_reward, best_expected_reward

    def context_fct(self, idx):
        return self.contexts[idx, :, :]


######################

class LogisticDataset(object):
    def __init__(self, info, theta_key, data_key):
        self.info = info
        self.theta = self.init_theta_star(theta_key)
        self.features, self.mean ,noise= self.generate_data(data_key)
        
        #jax.debug.print(self.theta.T@jnp.transpose(self.contexts ,(1,2,0) ) )
        #print(np.mean(self.theta.T@jnp.transpose(self.features ,(1,2,0) ) ))
        

    def init_theta_star(self, theta_key):
        theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1)) / np.sqrt(self.info.ctx_dim) 
        #theta /= jnp.linalg.norm(theta, ord=2)
        return theta

    def generate_data(self, data_key):
        key, subkey = jax.random.split(data_key)
        pool = jax.random.normal(subkey, shape=(50, self.info.ctx_dim))
        key, subkey = jax.random.split(key)
        indexes = jax.random.randint(subkey, minval=0, maxval=50, shape=(self.info.T, self.info.nb_arms))
        key, subkey = jax.random.split(key)
        contexts = pool[indexes] + self.info.context_noise * jax.random.normal(subkey, shape=(self.info.T, self.info.nb_arms, self.info.ctx_dim))
        mean = (contexts @ self.theta).squeeze()
        #print("print",mean.var())
        key, subkey = jax.random.split(key)
        noise = np.sqrt(1 + self.info.context_noise**2) * jax.random.normal(subkey, shape=(self.info.T,))
        return contexts, mean, noise



    def reward_fct(self, idx, data_key, action):
        key, subkey = jax.random.split(data_key)
        jax.debug.print("here {}",self.mean[idx, action] )
        expected_reward = 1 / (1 + jnp.exp(-  0.1*  self.mean[idx, action]     ) )
        reward = jnp.where(jax.random.bernoulli(subkey, p=expected_reward), 1, 0)
        best_expected_reward = 1 / (1 + jnp.exp(-jnp.max(self.mean[idx, :])))
        return key, reward, expected_reward, best_expected_reward

    def context_fct(self, idx):
        return self.features[idx, :, :]
    



#info=dict({"context_noise":0.1,"T":1000,"nb_arms":10,"ctx_dim":20})

# class Info(object):
#     def __init__(self):
#         self.T=1000
#         self.nb_arms=10
#         self.ctx_dim=20
#         self.context_noise=0.1

# info=Info()
# key, subkey = jax.random.split(203)
# Data=LogisticDataset(info,theta_key=key, data_key=key)
# print("finished")



"""
  
class LinearIllDataset(object):
    def __init__(self, info, theta_key, data_key):
        self.info = info
        self.theta = self.init_theta_star(theta_key)
        self.contexts, self.mean, self.noise = self.generate_data(data_key)

    def init_theta_star(self, theta_key):
        theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1))
        theta /= jnp.linalg.norm(theta, ord=2)
        return theta

    def generate_data(self, data_key):
        key, subkey = jax.random.split(data_key)
        z = jax.random.bernoulli(subkey, 0.9, shape=(self.info.T, self.info.nb_arms - 1))
        key, subkey = jax.random.split(key)
        eps = self.info.context_noise * jax.random.normal(subkey, shape=(self.info.T, self.info.nb_arms - 1, self.info.ctx_dim))
        key, subkey = jax.random.split(key)
        x1 = eps / jnp.linalg.norm(eps, ord=2, axis=2, keepdims=True)
        x2 = (self.theta.T[None, :, :] + eps) / jnp.linalg.norm(self.theta.T[None, :, :] + eps, ord=2, axis=2, keepdims=True)
        contexts = z[:, :, None] * x1 + (1 - z[:, :, None]) * x2
        contexts = jnp.concatenate((self.theta.T[None, :, :].repeat(self.info.T, axis=0), contexts), axis=1)
        mean = (contexts @ self.theta).squeeze()
        key, subkey = jax.random.split(key)
        noise = self.info.std_reward * jax.random.normal(subkey, shape=(self.info.T,))
        return contexts, mean, noise

    def reward_fct(self, idx, data_key, action):
        reward = self.mean[idx, action] + self.noise[idx]
        expected_reward = self.mean[idx, action].squeeze()
        best_expected_reward = jnp.max(self.mean[idx, :])
        raise ValueError(reward.shape, expected_reward.shape, best_expected_reward.shape)
        return data_key, reward, expected_reward, best_expected_reward

    def context_fct(self, idx):
        return self.contexts[idx].T
"""
class LinearIllDataset(object):
    def __init__(self, info, theta_key, data_key):
        self.info = info
        self.theta = self.init_theta_star(theta_key)
        self.contexts = self.generate_data(data_key)

    def init_theta_star(self, theta_key):
        theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1))
        theta /= jnp.linalg.norm(theta, ord=2)
        return theta

    def generate_data(self, data_key):
        key, subkey = jax.random.split(data_key)
        contexts = jax.random.normal(subkey, shape=(self.info.T, self.info.ctx_dim))
        contexts /= jnp.linalg.norm(contexts, ord=2, axis=1, keepdims=True)
        key, subkey = jax.random.split(key)
        arm_features = jax.random.normal(subkey, shape=(self.info.nb_arms, self.info.ctx_dim))
        arm_features /= jnp.linalg.norm(arm_features, ord=2, axis=1, keepdims=True)
        key, subkey = jax.random.split(key)
        Wx = jax.random.normal(subkey, shape=(self.info.ctx_dim, self.info.ctx_dim))
        Wx /= jnp.linalg.norm(Wx, ord=2, keepdims=True)
        key, subkey = jax.random.split(key)
        Wa = jax.random.normal(subkey, shape=(self.info.ctx_dim, self.info.ctx_dim))
        Wa /= jnp.linalg.norm(Wa, ord=2, keepdims=True)
        user_embeddings = contexts @ Wx 
        arm_embeddings = arm_features @ Wa
        contexts = user_embeddings[:, None, :] * arm_embeddings[None, :, :]
        #contexts /= jnp.linalg.norm(contexts, ord=2, axis=2, keepdims=True)
        return contexts

    def reward_fct(self, idx, data_key, action):
        ctx_vector = self.contexts[idx]
        mean_rewards = ctx_vector @ self.theta 
        key, subkey = jax.random.split(data_key)
        reward = mean_rewards[action][0] + self.info.std_reward * jax.random.normal(subkey, shape=(1,))[0]
        expected_reward = mean_rewards[action][0]
        best_expected_reward = jnp.max(mean_rewards)
        return key, reward, expected_reward, best_expected_reward

    def context_fct(self, idx):
        return self.contexts[idx]
    


#class LinearIllDataset(object):
#    def __init__(self, info, theta_key, data_key):
#        self.info = info
#        self.theta = self.init_theta_star(theta_key)
#        self.contexts, self.mean, self.noise = self.generate_data(data_key)
    
#    def modify_theta(self, key):
#        key, subkey = jax.random.split(key)
#        noise = 0.1 * jax.random.normal(subkey, shape=(1, self.info.ctx_dim))
#        theta_modif = self.theta.T + noise
#        theta_modif /= jnp.linalg.norm(theta_modif, axis=1, keepdims=True, ord=2)
#        return key, theta_modif
    
#    def generate_others(self, key):
#        key, subkey = jax.random.split(key)
#        x_others = jax.random.normal(subkey, shape=(self.info.nb_arms - 2, self.info.ctx_dim,))
#        x_others /= jnp.linalg.norm(x_others, axis=1, ord=2, keepdims=True)
#        return key, x_others

#    def generate_data(self, data_key):
#       key, subkey = jax.random.split(data_key)
#        subkey, theta_modified = self.modify_theta(subkey)
#        key, subkey = jax.random.split(subkey)
#        subkey, x_others = self.generate_others(subkey)
#        contexts = jnp.concatenate((self.theta.T, theta_modified, x_others), axis=0)
#        mean = (contexts @ self.theta).squeeze()
#        key, subkey = jax.random.split(key)
#        noise = self.info.std_reward * jax.random.normal(subkey, shape=(self.info.T,))
#        return contexts, mean, noise

#    def init_theta_star(self, theta_key):
#        theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1))
#        theta /= jnp.linalg.norm(theta, ord=2)
#        return theta

#    def reward_fct(self, idx, data_key, action):
#        reward = self.mean[action] + self.noise[idx]
#        expected_reward = self.mean[action].squeeze()
#        best_expected_reward = jnp.max(self.mean)
#        return data_key, reward, expected_reward, best_expected_reward

#    def context_fct(self, idx):
#        return self.contexts
    

# class LogisticDataset(object):
#     def __init__(self, info, theta_key, data_key):
#         self.info = info
#         self.theta = self.init_theta_star(theta_key)
#         self.features, self.mean = self.generate_data(data_key)

#     def generate_data(self, data_key):
#         contexts = jax.random.normal(data_key, shape=(self.info.T, self.info.nb_arms, self.info.ctx_dim))
#         contexts /= jnp.linalg.norm(contexts, ord=2, axis=2, keepdims=True)
#         mean = (contexts @ self.theta).squeeze()
#         return contexts, mean

#     def init_theta_star(self, theta_key):
#         theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1))
#         theta /= jnp.linalg.norm(theta, ord=2)
#         return theta

#     def reward_fct(self, idx, data_key, action):
#         key, subkey = jax.random.split(data_key)
#         expected_reward = 1 / (1 + jnp.exp(-self.mean[idx, action]) ) 
#         reward = jnp.where(jax.random.bernoulli(subkey, p=expected_reward), 1, 0)
#         best_expected_reward = 1 / (1 + jnp.exp(-jnp.max(self.mean[idx, :])))
#         return key, reward, expected_reward, best_expected_reward

#     def context_fct(self, idx):
#         return self.features[idx, :, :]


class LogisticIllDataset(object):
    def __init__(self, info, theta_key, data_key):
        self.info = info
        self.theta = self.init_theta_star(theta_key)
        self.contexts, self.mean, self.noise = self.generate_data(data_key)
    
    def modify_theta(self, key):
        key, subkey = jax.random.split(key)
        noise = 0.1 * jax.random.normal(subkey, shape=(1, self.info.ctx_dim))
        theta_modif = self.theta.T + noise
        theta_modif /= jnp.linalg.norm(theta_modif, axis=1, keepdims=True, ord=2)
        return key, theta_modif
    
    def generate_others(self, key):
        key, subkey = jax.random.split(key)
        x_others = jax.random.normal(subkey, shape=(self.info.T, self.info.nb_arms - 2, self.info.ctx_dim,))
        x_others /= jnp.linalg.norm(x_others, axis=2, ord=2, keepdims=True)
        return key, x_others

    def generate_data(self, data_key):
        key, subkey = jax.random.split(data_key)
        subkey, theta_modified = self.modify_theta(subkey)
        key, subkey = jax.random.split(subkey)
        subkey, x_others = self.generate_others(subkey)
        contexts = jnp.concatenate((self.theta.T[None, :, :], theta_modified[None, :, :], x_others), axis=0)
        mean = (contexts @ self.theta).squeeze()
        key, subkey = jax.random.split(key)
        noise = self.info.std_reward * jax.random.normal(subkey, shape=(self.info.T,))
        return contexts, mean, noise

    def init_theta_star(self, theta_key):
        theta = jax.random.normal(theta_key, shape=(self.info.ctx_dim, 1))
        theta /= jnp.linalg.norm(theta, ord=2)
        return theta

    def reward_fct(self, idx, data_key, action):
        key, subkey = jax.random.split(data_key)
        expected_reward =1 / (1 + jnp.exp(-self.mean[idx, action].squeeze()))
        reward = jnp.where(jax.random.bernoulli(subkey, p=expected_reward), 1, 0)
        best_expected_reward = 1 / (1 + jnp.exp(-jnp.max(self.mean[idx, :]).squeeze()))
        return key, reward, expected_reward, best_expected_reward

    def context_fct(self, idx):
        return self.contexts[idx, :, :]

class YahooEnv(object):
    def __init__(self, info, theta_key, data_key):
        self.info = info
        self.displayed_arms, self.rewards, self.user_features, self.pool_indexes, self.features = self.init_dataset(3000000, data_key)

    def init_dataset(self, limit_size, data_key):
        if exists('dataset/yahoo.pkl'):
            displayed_arms, rewards, user_features, pool_indexes, features = tuple(map(lambda x: jnp.array(x), pkl.load(open('dataset/yahoo_numpy.pkl', "rb"))))
            indexes = jax.random.shuffle(data_key, jnp.arange(len(rewards)), axis=0)
            displayed_arms = jnp.array(displayed_arms)[indexes]
            rewards = jnp.array(rewards)[indexes]
            user_features = jnp.array(user_features)[indexes]
            pool_indexes = jnp.array(pool_indexes)[indexes]
            print('Number of articles:', rewards.shape[0])
            return displayed_arms, rewards, user_features, pool_indexes, features
        else:
            return self.load_dataset(limit_size)

    def load_dataset(self, limit_size):
        features, articles = [], []
        displayed_arms, rewards, user_features, pool_indexes = [], [], [], []
        files = ('dataset/yahoo.pkl')
        with fileinput.input(files=files) as f:
            for line in tqdm(f):
                cols = line.split()
                if (len(cols) - 10) % 7 != 0:
                    pass
                else:
                    pool_idx = []
                    pool_ids = []
                    for i in range(10, len(cols) - 6, 7):
                        id = cols[i][1:]
                        if id not in articles:
                            articles.append(id)
                            features.append([float(x[2:]) for x in cols[i + 1: i + 7]])
                        pool_idx.append(articles.index(id))
                        pool_ids.append(id)
                    user_feature = [float(x[2:]) for x in cols[4:10]]
                    if len(pool_idx) == 19:
                        user_features.append(user_feature)
                        pool_indexes.append(pool_idx)
                        displayed_arms.append(pool_ids.index(cols[1]))
                        rewards.append(int(cols[2]))
                    if limit_size != None and len(rewards) == limit_size:
                        break
        print('End of load')
        features = jnp.array(features)
        displayed_arms = jnp.array(displayed_arms)
        rewards = jnp.array(rewards)
        user_features = jnp.array(user_features)
        pool_indexes = jnp.array(pool_indexes)
        print('End of save')
        pkl.dump((displayed_arms, rewards, user_features, pool_indexes, features), open('dataset/yahoo.pkl', "wb"))
        print('End of load_dataset')
        print('Number of articles:', rewards.shape[0])
        return displayed_arms, rewards, user_features, pool_indexes, features
        
    def reward_fct(self, idx, data_key, action):
        reward = self.rewards[idx]
        return data_key, reward, reward, 1

    def context_fct(self, idx):
        acticle_features = self.features[self.pool_indexes[idx]]
        user_features = jnp.tile(self.user_features[idx], (acticle_features.shape[0], 1))
        contexts = jnp.concatenate((user_features, acticle_features), axis=1)
        return self.displayed_arms[idx], contexts

