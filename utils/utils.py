import jax
import jax.numpy as jnp
from typing import Sequence
from flax import linen as nn
import numpy as np
from functools import partial
from jax.nn import softplus

class MLP(nn.Module):
    features: Sequence[int]
    ctx_dim: int
    logistic_activation : bool

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1] , use_bias=False)(x)
        return x
    
    def __hash__(self):
        return id(self)
    
    @partial(jax.jit, static_argnums=(0,))
    def pytree_size(self):
        layers = [self.ctx_dim] + self.features
        size = sum([(layers[i]+1) * layers[i+1] for i in range(len(layers) - 1)]) - 1
        return size
    

class UtilsVector(object):
    def __init__(self, info, agent_key):
        self.agent_key = agent_key
        self.info = info
        self.utils_vector= {
            "lmcts": self.init_LMC,
            "ts": self.init_LinTS_LinUCB,
            "ucb": self.init_LinTS_LinUCB,
            "vits": self.init_VITS,
            "rvits": self.init_RVITS,
            "vts": self.init_VTS,
            "random": None
        }[self.info.agent_name]()

    def slice_vector(self, vector_old, idx, vector_new):
        in_shape = list(vector_old.shape)
        in_shape[0] = idx
        out_shape = list(vector_old.shape)
        out_shape[0] = vector_old.shape[0] - idx
        vector_sliced = jax.lax.dynamic_slice(vector_old, in_shape, out_shape)
        return jnp.concatenate((vector_new, vector_sliced))

    def set(self, utils_vector, idx=None):
        if idx == None or self.info.agent_name == "ts" or self.info.agent_name == "ucb" or self.info.agent_name == "random":
            self.utils_vector = utils_vector
        else:
            if self.info.agent_name == "vits":
                features_new, labels_new, mean, cov_semi, cov_semi_inv  = utils_vector
                features_old, labels_old, _, _, _ = self.utils_vector
                features = self.slice_vector(features_old, idx, features_new)
                labels = self.slice_vector(labels_old, idx, labels_new)
                self.utils_vector = features, labels, mean, cov_semi, cov_semi_inv

            elif self.info.agent_name == "rvits":
                features_new, labels_new, mean, cov  = utils_vector
                features_old, labels_old, _, _, = self.utils_vector
                features = self.slice_vector(features_old, idx, features_new)
                labels = self.slice_vector(labels_old, idx, labels_new)
                self.utils_vector = features, labels, mean, cov

            elif self.info.agent_name == "lmcts":
                features_new, labels_new, theta = utils_vector
                features_old, labels_old, _ = self.utils_vector
                features = self.slice_vector(features_old, idx, features_new)
                labels = self.slice_vector(labels_old, idx, labels_new)
                self.utils_vector = features, labels, theta
            elif self.info.agent_name == "vts":
                features_new, actions_new, labels_new, alpha, beta, gamma, theta, Sigma, Sigma_inv = utils_vector
                features_old, actions_old, labels_old, _, _,  _, _, _,  _  = self.utils_vector
                features = self.slice_vector(features_old, idx, features_new)
                labels = self.slice_vector(labels_old, idx, labels_new)
                actions = self.slice_vector(actions_old, idx, actions_new)
                self.utils_vector = features,  actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv,
            else:
                raise NotImplementedError

    def get(self, idx=None):
        if idx == None or self.info.agent_name == "ts" or self.info.agent_name == "ucb" or self.info.agent_name == "random":
            return self.utils_vector
        else:
            if self.info.agent_name == "vits":
                features, labels, mean, cov_semi, cov_semi_inv = self.utils_vector
                features_sliced = jax.lax.dynamic_slice(features, (0, 0), (idx, features.shape[1]))
                labels_sliced = jax.lax.dynamic_slice(labels, (0,), (idx,))
                return features_sliced, labels_sliced, mean, cov_semi, cov_semi_inv
            
            if self.info.agent_name == "rvits":
                features, labels, mean, cov = self.utils_vector
                features_sliced = jax.lax.dynamic_slice(features, (0, 0), (idx, features.shape[1]))
                labels_sliced = jax.lax.dynamic_slice(labels, (0,), (idx,))
                return features_sliced, labels_sliced, mean, cov
            
            elif self.info.agent_name == "lmcts" or self.info.agent_name == 'lmcts_new':
                features, labels, theta = self.utils_vector
                features_sliced = jax.lax.dynamic_slice(features, (0, 0), (idx, features.shape[1]))
                labels_sliced = jax.lax.dynamic_slice(labels, (0,), (idx,))
                return features_sliced, labels_sliced, theta
            
            elif self.info.agent_name == "vts":
                (features, actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv) = self.utils_vector
                features_sliced = jax.lax.dynamic_slice(features, (0, 0), (idx, features.shape[1]))
                actions_sliced = jax.lax.dynamic_slice(actions, (0,), (idx,))
                labels_sliced = jax.lax.dynamic_slice(labels, (0,), (idx,))
                return features_sliced, actions_sliced, labels_sliced, alpha, beta, gamma, theta, Sigma, Sigma_inv
            else:
                raise NotImplementedError

    def init_LinTS_LinUCB(self):
        bt = jnp.zeros((self.info.d, 1))
        cov = jnp.eye(self.info.d) / self.info.lbd
        return bt, cov

    def init_LMC(self):
        self.model = MLP(self.info.model.layers, self.info.ctx_dim, self.info.model.logistic_activation)
        self.dimension = self.model.pytree_size()
        self.agent_key, subkey = jax.random.split(self.agent_key)
        features = jnp.zeros((self.info.T, self.info.ctx_dim))
        labels = jnp.zeros((self.info.T))
        theta = self.model.init(subkey, features)
        return features, labels, theta
    
    def init_VITS(self):
        self.model = MLP(self.info.model.layers, self.info.ctx_dim,self.info.model.logistic_activation)
        self.dimension = self.model.pytree_size()
        features = jnp.zeros((self.info.T, self.info.ctx_dim))
        labels = jnp.zeros((self.info.T))
        mean = jnp.zeros((1, self.dimension))
        cov_semi = jnp.eye(self.dimension) / jnp.sqrt(self.info.eta)
        cov_semi_inv = jnp.eye(self.dimension) * jnp.sqrt(self.info.eta)
        return features, labels, mean, cov_semi, cov_semi_inv
    
    def init_RVITS(self):
        self.model = MLP(self.info.model.layers, self.info.ctx_dim, self.info.model.logistic_activation)
        self.dimension = self.model.pytree_size()
        features = jnp.zeros((self.info.T, self.info.ctx_dim))
        labels = jnp.zeros((self.info.T))
        mean = jnp.zeros((1, self.dimension))
        cov = jnp.eye(self.dimension) / (self.info.eta * self.info.lbd)
        return features, labels, mean, cov

    def init_VTS(self):
        features = jnp.zeros((self.info.T, self.info.ctx_dim))
        actions = jnp.zeros((self.info.T))
        labels = jnp.zeros((self.info.T))
        self.gamma_0 = self.info.vts.gamma_0 * jnp.ones((self.info.nb_arms, self.info.vts.nb_mixtures))
        self.alpha_0 = self.info.vts.alpha_0 * jnp.ones((self.info.nb_arms, self.info.vts.nb_mixtures))
        self.beta_0 = self.info.vts.beta_0 * jnp.ones((self.info.nb_arms, self.info.vts.nb_mixtures))
        self.Sigma_0 = jnp.tile(self.info.vts.sigma_0 * jnp.identity(self.info.ctx_dim), (self.info.nb_arms, self.info.vts.nb_mixtures, 1, 1))
        self.Sigma_0_inv = jnp.tile((1 / self.info.vts.sigma_0) * jnp.identity(self.info.ctx_dim), (self.info.nb_arms, self.info.vts.nb_mixtures, 1, 1),)
        self.theta_0 = jnp.tile(jnp.expand_dims(self.info.vts.theta_0 * jnp.arange(0, self.info.vts.nb_mixtures, dtype=jnp.float32), (0, 2)), (self.info.nb_arms, 1, self.info.ctx_dim))
        return features, actions, labels, self.alpha_0, self.beta_0, self.gamma_0, self.theta_0, self.Sigma_0, self.Sigma_0_inv


def binary_logistic_loss(label: int, logit: float) -> float:
  # Softplus is the Fenchel conjugate of the Fermi-Dirac negentropy on [0, 1].
  # softplus = proba * logit - xlogx(proba) - xlogx(1 - proba),
  # where xlogx(proba) = proba * log(proba).
  # Use -log sigmoid(logit) = softplus(-logit)
  # and 1 - sigmoid(logit) = sigmoid(-logit).
  logit = nn.sigmoid(logit)
  return softplus(jnp.where(label, -logit, logit))
