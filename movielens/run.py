# Imports and defaults
import numpy as np
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from torch.autograd.functional import hessian
import pandas as pd
import seaborn as sns
import wandb
from torch.distributions.multivariate_normal import MultivariateNormal
import argparse
os.environ["WANDB_MODE"] = "offline"


class ThompsonSampling(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, eta, lbd):
        self.eta = eta
        self.lbd = lbd
        self.device = device
        self.dimension = dimension
        self.V_inv =  eta * cov_prior
        self.bt = torch.diag(1/torch.diag(self.V_inv)) @ mean_prior

    def get_config(self):
        return {'eta': self.eta, 'lbd': self.lbd, 'dimension': self.dimension, 'algorithm': 'TS'}
        
    def sample_posterior(self):
        mean = self.V_inv @ self.bt
        cov = self.V_inv / self.eta
        return MultivariateNormal(mean, covariance_matrix=cov).sample()
        
    def reward(self, user):
        theta = self.sample_posterior()
        return user.dot(theta).item()
        
    def update(self, user, action, reward):
        self.bt += reward * user
        omega = self.V_inv @ user
        self.V_inv -= omega[:, None] @ omega[None, :] / (1 + omega.dot(user))
    
    
class VITS(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, eta, lbd, is_linear, h, nb_updates):
        self.eta = eta
        self.device = device
        self.lbd = lbd
        self.dimension = dimension
        self.users = torch.empty((0, dimension)).to(self.device)
        self.rewards = torch.empty((0,)).to(self.device)
        self.linear = is_linear
        self.h = h
        self.nb_updates = nb_updates
        self.mean = torch.tensor(mean_prior, dtype=torch.float32).to(self.device)
        self.cov_semi = torch.diag(torch.sqrt(torch.diag(torch.tensor(cov_prior, dtype=torch.float32)))).to(self.device)
        self.cov_semi_inv = torch.diag(1/torch.sqrt(torch.diag(torch.tensor(cov_prior, dtype=torch.float32)))).to(self.device)
        
    def get_config(self):
        return {'eta': self.eta, 'lbd': self.lbd, 'dimension': self.dimension, 'h': self.h,
                'nb_updates': self.nb_updates, 'algorithm': 'VITS'}

    def sample_posterior(self):
        eps = torch.normal(0, 1, size=(self.dimension,)).to(self.device)
        theta = self.mean + self.cov_semi @ eps
        return theta
        
    def reward(self, user):
        theta = self.sample_posterior()
        return user.dot(theta).item()
    
    def potential(self, theta):
        if self.linear:
            data_term = torch.sum(torch.square(self.users @ theta - self.rewards))
            regu = self.lbd * theta.dot(theta)
            return self.eta * (data_term + regu) / 2
        else:
            raise ValueError('to implement')
    
    def compute_gradient_hessian(self):
        theta = self.sample_posterior()
        theta.requires_grad = True
        gradient = grad(self.potential(theta), theta)[0]
        hessian_matrix = hessian(self.potential, theta)
        del theta
        return gradient, hessian_matrix
    
    def update_cov(self, h, hessian_matrix):
        cov_semi = (torch.eye(self.dimension).to(self.device) - h * hessian_matrix) @ self.cov_semi + h * self.cov_semi_inv.T
        cov_semi_inv = torch.linalg.pinv(cov_semi)
        return cov_semi, cov_semi_inv
        
    def update(self, user, action, reward):
        self.users = torch.cat([self.users, torch.tensor(user, dtype=torch.float32)[None, :].to(self.device)])
        self.rewards = torch.cat([self.rewards, torch.tensor([reward], dtype=torch.float32).to(self.device)])
        h = self.h / (len(self.rewards) * self.eta)
        for _ in range(self.nb_updates):
            gradient, hessian_matrix = self.compute_gradient_hessian()
            self.mean -= h * gradient
            self.cov_semi, self.cov_semi_inv = self.update_cov(h, hessian_matrix)
            del gradient, hessian_matrix



class Langevin(object):
    def __init__(self, dimension, mean_prior, cov_prior, device, eta, lbd, is_linear, h, nb_updates):
        self.eta = eta
        self.lbd = lbd
        self.device = device
        self.dimension = dimension
        self.users = torch.empty((0, dimension)).to(self.device)
        self.rewards = torch.empty((0,)).to(self.device)
        self.linear = is_linear
        self.h = h
        self.nb_updates = nb_updates
        self.theta = torch.tensor(mean_prior, dtype=torch.float32).to(self.device)

    def get_config(self):
        return {'eta': self.eta, 'lbd': self.lbd, 'dimension': self.dimension, 'h': self.h,
                'nb_updates': self.nb_updates, 'algorithm': 'lmcts'}

    def reward(self, user):
        return user.dot(self.theta).item()
    
    def potential(self, theta):
        if self.linear:
            data_term = torch.sum(torch.square(self.users @ theta - self.rewards))
            regu = self.lbd * theta.dot(theta)
            return (data_term + regu) / 2
        else:
            raise ValueError('to implement')
    
    def compute_gradient(self):
        self.theta.requires_grad = True
        gradient = grad(self.potential(self.theta), self.theta)[0]
        self.theta.requires_grad = False
        return gradient
    
    def update(self, user, action, reward):
        self.users = torch.cat([self.users, torch.tensor(user, dtype=torch.float32).to(self.device)[None, :]])
        self.rewards = torch.cat([self.rewards, torch.tensor([reward], dtype=torch.float32).to(self.device)])
        h = self.h / len(self.rewards) 
        for _ in range(self.nb_updates):
            gradient = self.compute_gradient()
            self.theta += -h * gradient + torch.normal(0, np.sqrt(2 * h / self.eta), size=gradient.shape).to(self.device)
            del gradient
    
class MovieLens(object):
    def __init__(self, dimension, regularisation, nb_iters, device):
        self.dimension = dimension
        self.device = device
        self.regularisation = regularisation
        self.nb_iters = nb_iters
        if os.path.exists(f'movielens_data_{dimension}.pkl'):
            self.users, self.movies = pkl.load(open(f'movielens_data_{dimension}.pkl', 'rb'))
            np.random.shuffle(self.users)
            np.random.shuffle(self.movies) 
        else:
            self.users, self.movies = self.load()
            pkl.dump((self.users, self.movies), open(f'movielens_data_{dimension}.pkl', 'wb'))
        self.users = torch.tensor(self.users, dtype=torch.float32).to(self.device)
        self.movies = torch.tensor(self.movies, dtype=torch.float32).to(self.device)
            
    def load(self):
        data = np.loadtxt("ratings.txt")
        data = data[:, : 3].astype(int)
        num_users = data[:, 0].max()
        num_movies = data[:, 1].max()
        data[:, : 2] -= 1
        M = np.zeros((num_users, num_movies))
        M[data[:, 0], data[:, 1]] = data[:, 2]
        W = np.zeros((num_users, num_movies))
        W[data[:, 0], data[:, 1]] = 1
        ndx = np.random.permutation(num_users)
        M = M[ndx, :]
        W = W[ndx, :]
        users = 2 * np.random.rand(num_users, self.dimension) - 1
        movies = 2 * np.random.rand(num_movies, self.dimension) - 1
        for iter in range(self.nb_iters):
            for i in range(num_users):
                sel = np.flatnonzero(W[i, :])
                G = movies[sel, :].T.dot(movies[sel, :]) + self.regularisation * np.eye(self.dimension)
                users[i, :] = np.linalg.solve(G, movies[sel, :].T.dot(M[i, sel]))
            for j in range(num_movies):
                sel = np.flatnonzero(W[:, j])
                G = users[sel, :].T.dot(users[sel, :]) + self.regularisation * np.eye(self.dimension)
                movies[j, :] = np.linalg.solve(G, users[sel, :].T.dot(M[sel, j]))
            print("%.3f " % (np.linalg.norm(W * (M - users.dot(movies.T))) / np.sqrt(W.sum())), end="")
            print()
        np.random.shuffle(users)
        np.random.shuffle(movies)
        return users, movies
    
    def sample_user(self):
        return self.users[np.random.choice(range(len(self.users)))]
        
    def evaluate(self, user, action):
        rewards = self.movies @ user
        expected_reward = rewards[action]
        best_expected_reward = rewards.max()
        reward = expected_reward + torch.normal(0, 1, size=(1,)).to(self.device)[0]
        return reward, expected_reward, best_expected_reward
    
    def choose_movie(self, user, agents):
        rewards = [agent.reward(user) for agent in agents]
        return np.argmax(rewards)
    
    def compute(self, Agent, hyperparameters, T, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.movies = self.movies[np.random.choice(range(len(self.movies)), size=(100,)), :]
        mean_prior = torch.mean(self.movies, axis=0)
        cov_prior = torch.diag(torch.var(self.movies, axis=0))
        agents = [Agent(self.dimension, mean_prior, cov_prior, self.device, *hyperparameters) for _ in range(len(self.movies))]
        wandb.init(config=agents[0].get_config(), project='movielens_lmcts2')
        cumulative_regret = torch.zeros((T,))
        for t in range(T):
            user = self.sample_user()
            action = self.choose_movie(user, agents)
            reward, expected_reward, best_expected_reward = self.evaluate(user, action)
            agents[action].update(user, action, reward)
            cumulative_regret[t] = cumulative_regret[t-1] + best_expected_reward - expected_reward
            wandb.log({'cum_regret': cumulative_regret[t]})
        wandb.finish()
        return cumulative_regret
    

parser = argparse.ArgumentParser()
parser.add_argument('--seed')
parser.add_argument('--algo')
parser.add_argument('--lr')
if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[SYSTEM], device: {device}')
    dimension = 5
    data = MovieLens(dimension, 1, 50, device)

    eta_list = [100]
    lbd_list = [0.1]

    T = 5000

    if args.algo == 'ts':
        algo_name = 'TS'
        algo = ThompsonSampling
        hyperparameter = lambda eta, lbd : (eta, lbd)
    elif args.algo == 'vits':
        algo_name = 'VITS'
        algo = VITS
        hyperparameter = lambda eta, lbd : (eta, lbd, True, float(args.lr), 10)
    elif args.algo == 'lmcts':
        algo_name = 'LMC-TS'
        algo = Langevin
        hyperparameter = lambda eta, lbd : (eta, lbd, True, float(args.lr), 100)
    else:
        raise ValueError(args.algo)

    df = pd.DataFrame()
    for eta in eta_list:
        for lbd in lbd_list:
            row = pd.DataFrame({'seed': args.seed,
                                'legend': f'{algo_name} - eta: {eta} - lambda: {lbd}',
                                'step': range(T),
                                'cum_regret': data.compute(algo, hyperparameter(eta, lbd), T, dimension)})
            df = pd.concat([df, row], ignore_index=True)

    #pkl.dump(df, open('resultat.pkl', 'wb'))
    #plt.style.use('seaborn')
    #sns.lineplot(data=df, x='step', y='cum_regret', hue='legend')
    #plt.savefig()
