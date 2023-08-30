import numpy as np
import jax
from jax import numpy as jnp
from jax import grad, jacfwd
import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict
from jax.lax import dynamic_slice
from functools import partial
from utils.utils import binary_logistic_loss

class VITS(object):
    def __init__(self, info, utils_object):
        self.info = info
        self.utils_object = utils_object
        self.grad_function = grad(self.potential, argnums=0)
        self.hessian_function = jacfwd(grad(self.potential, argnums=0))
        self.layers = [self.info.ctx_dim] + self.info.model.layers
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
        features, labels, mean, cov_semi, cov_semi_inv = utils_vector
        key, theta = self.sample(key, mean, cov_semi)
        theta_nn = self.fill(theta)
        rewards = self.utils_object.model.apply(theta_nn, context)
        utils_vector = (features, labels, mean, cov_semi, cov_semi_inv)
        return key, utils_vector, rewards.argmax()
    
    def potential(self, theta, features, labels):
        theta_nn = self.fill(theta)
        prediction = self.utils_object.model.apply(theta_nn, features)
        data_term = binary_logistic_loss(labels,prediction) if self.info.env == 'logistic' else jnp.square(prediction - labels)
        return data_term.squeeze()
       
    def sample(self, key, mean, cov_semi):
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, shape=(self.utils_object.dimension, 1))
        theta = (mean.T + cov_semi @ eps).squeeze()
        return key, theta
    
    def get_gradient(self, theta, features, labels):
        regularization_grad = 2 * self.info.lbd * theta
        data_grad = jnp.sum(jax.vmap(self.grad_function, in_axes=(None, 0, 0))(theta, features, labels), axis=0)
        return self.info.eta * (regularization_grad + data_grad)
    
    def get_hessian(self, theta, cov_semi_inv, mean, gradient, features, labels):
        if self.info.vits.hessian_free:
            return (cov_semi_inv @ cov_semi_inv.T) @ (theta - mean).T @ gradient[None, :]
        else:
            regularization_hessian = 2 * self.info.lbd * jnp.eye(self.utils_object.dimension)
            data_hessian = jnp.sum(jax.vmap(self.hessian_function, in_axes=(None, 0, 0))(theta, features, labels), axis=0)
            return self.info.eta *  (regularization_hessian + data_hessian)

    def update_mean(self, mean, gradient, h):
        return mean - h * gradient
    
    def updatye_cov(self, cov_semi, cov_semi_inv, hessian, h):
        cov_semi = (jnp.eye(self.utils_object.dimension) - h * hessian) @ cov_semi + h * cov_semi_inv.T
        if self.info.vits.approx:
            cov_semi_inv = cov_semi_inv @ (jnp.eye(self.utils_object.dimension) - h * (jnp.matmul(cov_semi_inv.T , cov_semi_inv) - hessian))
        else:
            cov_semi_inv = jnp.linalg.inv(cov_semi)
        return cov_semi, cov_semi_inv
  
    def update_law(self, idx, features, labels,key, mean, cov_semi, cov_semi_inv):
        key, subkey = jax.random.split(key)
        _, theta = self.sample(subkey, mean, cov_semi)
        gradient = self.get_gradient(theta, features, labels)
        hessian = self.get_hessian(theta, cov_semi_inv, mean, gradient, features, labels)
        h = self.info.vits.step_size / features.shape[0]
        mean = self.update_mean(mean, gradient, h)
        cov_semi, cov_semi_inv = self.update_cov(cov_semi, cov_semi_inv, hessian, h)
        return (key, mean, cov_semi, cov_semi_inv)

    def update_fct(self, key, context, action, reward, utils_vector):
        features, labels, mean, cov_semi, cov_semi_inv = utils_vector
        features = features.at[-1].set(context[action, :])
        labels = labels.at[-1].set(reward)       

        key, mean, cov_semi, cov_semi_inv = jax.lax.fori_loop(
            0,
            self.info.vits.num_updates,
            lambda i, v: self.update_law(i, features, labels, *v),
            (key, mean, cov_semi, cov_semi_inv))
        return key, (features, labels, mean, cov_semi, cov_semi_inv)