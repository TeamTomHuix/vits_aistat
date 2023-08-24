import numpy as np
import jax
from jax import numpy as jnp
from jax import grad, jacfwd
import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict
from jax.lax import dynamic_slice
from functools import partial
from utils.utils import binary_logistic_loss
from copy import deepcopy

class RVITS(object):
    def __init__(self, info, utils_object):
        raise NotImplementedError
        self.info = info
        self.utils_object = utils_object

        if self.info["env"]== "logisticbis" or self.info['env'] == "logistic":
            self.grad_function = grad(self.potential_logistic, argnums=0)
            self.hessian_function = jacfwd(grad(self.potential_logistic, argnums=0))
        else :
            self.grad_function = grad(self.potential, argnums=0)
            self.hessian_function = jacfwd(grad(self.potential, argnums=0))

        self.layers = [self.info['ctx_dim']] + self.info['layers']
        self.slicing_funtions = {f'Dense_{i}': self.create_slice_function(i) for i in range(len(self.layers) - 1)}

    def create_slice_function(self, i):
        index = sum([(self.layers[j] +1) * self.layers[j+1] for j in range(i)])
        if i == len(self.layers) - 2:
            return jax.jit(lambda t: {'kernel': dynamic_slice(t, (index,), (self.layers[i] * self.layers[i+1],)).reshape((self.layers[i], self.layers[i+1]))})
        else:
            return jax.jit(lambda t: {
                'kernel': dynamic_slice(t, (index,), (self.layers[i] * self.layers[i+1],)).reshape((self.layers[i], self.layers[i+1])),
                'bias': dynamic_slice(t, (index + (self.layers[i] * self.layers[i+1]),), (self.layers[i+1],))})

    @partial(jax.jit, static_argnums=(0,))
    def fill(self, theta):
        return FrozenDict({'params': {key: fct(theta) for key, fct in self.slicing_funtions.items()}})

    def choice_fct(self, key, context, utils_vector):
        features, labels, mean, cov = utils_vector
        key, theta = self.sample(key, mean, cov)
        theta_nn = self.fill(theta)
        rewards = self.utils_object.model.apply(theta_nn, context)
        utils_vector = (features, labels, mean, cov)
        return key, utils_vector, rewards.argmax()
    
    def potential(self, theta, features, labels):
        theta_nn = self.fill(theta)
        prediction = self.utils_object.model.apply(theta_nn, features)
        data_term = jnp.square(prediction - labels)
        return data_term.squeeze()
    
    def potential_logistic(self, theta, features, labels):
        theta_nn = self.fill(theta)
        prediction = self.utils_object.model.apply(theta_nn, features)
        data_term=binary_logistic_loss(labels,prediction)
        return data_term.squeeze()
       
    def sample(self, key, mean, cov):
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, shape=(self.utils_object.dimension, 1))
        theta = (mean.T + jnp.linalg.inv(cov) @ eps).squeeze()
        return key, theta
    
    def update_law(self,  idx, features, labels, key, mean, prev_mean, cov, prev_cov_inv):
        def compute_loss_gradient(key, mean, cov):
            key, theta = self.sample(key, mean, cov)
            gradient =  jnp.sum(jax.vmap(self.grad_function, in_axes=(None, 0, 0))(theta, features, labels), axis=0)
            gradient += 2 * self.info["lambda"] * theta
            return key, self.info["eta"] * gradient
        
        def compute_loss_hessian(key, mean, cov):
            key, theta = self.sample(key, mean, cov)
            hessian = jnp.sum(jax.vmap(self.hessian_function, in_axes=(None, 0, 0))(theta, features, labels), axis=0)
            hessian += 2 * self.info["lambda"] * jnp.eye(self.utils_object.dimension)
            return key, self.info["eta"] *  hessian
        
        key, loss_gradient = compute_loss_gradient(key, mean, cov)
        key, loss_hessian = compute_loss_hessian(key, mean, cov)
        gradient_mean = prev_cov_inv @ (mean - prev_mean).T + (loss_gradient) / (2 * self.info["step_size"])
        gradient_cov = 0.5 * (prev_cov_inv - jnp.linalg.inv(cov) + loss_hessian)

        # Update the parameters
        h = self.info["step_size"] / features.shape[0]

        mean = mean - h * gradient_mean
        M = (jnp.eye(self.utils_object.dimension) - 2 * h * gradient_cov)
        cov = M.T @ cov  @ M

        return (key, mean, prev_mean, cov, prev_cov_inv)

    def update_fct(self, key, context, action, reward, utils_vector):
        features, labels, mean, cov = utils_vector
        features = features.at[-1].set(context[action, :])
        labels = labels.at[-1].set(reward)       

        prev_mean = deepcopy(mean)
        prev_cov_inv = jnp.linalg.inv(cov)

        key, mean, cov, _ = jax.lax.fori_loop(
            0,
            self.info["num_updates"],
            lambda i, v: self.update_law(i, features, labels, *v),
            (key, mean, prev_mean, cov, prev_cov_inv))

        return key, (features, labels, mean, cov)
