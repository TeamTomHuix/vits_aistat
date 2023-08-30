import jax
from jax import numpy as jnp
from jax import grad
import numpy as np
from functools import partial
from flax.core.frozen_dict import FrozenDict
from jax.lax import dynamic_slice
from utils.utils import binary_logistic_loss

class LMCTS(object):
    def __init__(self, info, utils_object):
        self.info = info
        self.utils_object = utils_object
        self.grad_function = grad(self.potential, argnums=0)
        self.layers = [self.info.ctx_dim] + self.info.model.layers
        self.slicing_funtions = {f'Dense_{i}': self.create_slice_function(i) for i in range(len(self.layers) - 1)}

    def choice_fct(self, key, context, utils_vector):
        features, labels, theta = utils_vector
        rewards = self.utils_object.model.apply(theta, context)
        return key, (features, labels, theta), rewards.argmax()
    
    def potential(self, theta, features, labels):
        def logistic_loss(feature, label):
            prediction = self.utils_object.model.apply(theta, feature)
            data_term = binary_logistic_loss(label,prediction)
            return data_term
        def square_loss(feature, label):
            prediction = self.utils_object.model.apply(theta, feature)
            data_term = jnp.square(prediction - label)
            return data_term
        loss_fct = logistic_loss if self.info.env == 'logistic' else square_loss
        regularization_term = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(theta))
        data_term = jnp.sum(jax.vmap(loss_fct)(features, labels))
        return data_term + self.info.lbd * regularization_term
    
    def create_slice_function(self, i):
        index = sum([(self.layers[j] +1) * self.layers[j+1] for j in range(i)])
        if i == len(self.layers) - 2:
            return jax.jit(lambda t: {'kernel': dynamic_slice(t, (index,), (self.layers[i] * self.layers[i+1],)).reshape((self.layers[i], self.layers[i+1]))})
        else:
            return jax.jit(lambda t: {
                'kernel': dynamic_slice(t, (index,), (self.layers[i] * self.layers[i+1],)).reshape((self.layers[i], self.layers[i+1])),
                'bias': dynamic_slice(t, (index + (self.layers[i] * self.layers[i+1]),), (self.layers[i+1],))})

    @partial(jax.jit, static_argnums=(0,))
    def fill(self, vector):
        return FrozenDict({'params': {key: fct(vector) for key, fct in self.slicing_funtions.items()}})

    def update_law(self, idx, features, labels, step_size, noise, theta):
        grad_theta = self.grad_function(theta, features, labels)
        noise_tree = self.fill(noise[idx])
        theta = jax.tree_util.tree_map(lambda t, g, n: t - step_size * g + n, theta, grad_theta, noise_tree)
        return theta

    def update_fct(self, key, context, action, reward, utils_vector):
        features, labels, theta = utils_vector
        features = features.at[-1].set(context[action, :])
        labels = labels.at[-1].set(reward)
        step_size = self.info.lmcts.step_size / features.shape[0]

        key, subkey = jax.random.split(key)
        noises = np.sqrt(2 * step_size / self.info.eta) * jax.random.normal(subkey, shape=(self.info.lmcts.num_updates, self.utils_object.dimension))

        theta = jax.lax.fori_loop(
            0,
            self.info.lmcts.num_updates,
            lambda i, theta: self.update_law(i, features, labels, step_size, noises, theta),
            theta,
        )
        return key, (features, labels, theta)
