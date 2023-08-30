import jax
import jax.numpy as jnp

class Random(object):
    def __init__(self, info, utils_object):
        self.info = info
        self.utils_object = utils_object

    def choice_fct(self, key, context, utils_vector):
        key, subkey = jax.random.split(key)
        action = jax.random.randint(subkey, (1,), 0, self.info.nb_arms)
        return key, utils_vector, action

    def update_fct(self, key, context, action, reward, utils_vector):
        pass
        return key, utils_vector