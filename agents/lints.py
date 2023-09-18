import jax
import jax.numpy as jnp

class LinTS(object):
    def __init__(self, info, utils_object):
        self.info = info
        self.utils_object = utils_object

    def choice_fct(self, key, context, utils_vector):
        bt, cov = utils_vector
        key, subkey = jax.random.split(key)
        mean = cov @ bt
        theta = jax.random.multivariate_normal(subkey, mean.squeeze(), cov / self.info.eta)
        rewards = context.squeeze() @ theta.squeeze()
        return key, (bt, cov), rewards.argmax()
    
    def compute_cond_number(self, utils_vector):
        _, cov = utils_vector
        return jnp.linalg.cond(jnp.linalg.inv(cov))

    def update_fct(self, key, context, action, reward, utils_vector):
        bt, cov = utils_vector
        selected_context = context[action, :]
        bt += (reward * selected_context).reshape((bt.shape[0], 1))
        omega = (cov @ selected_context).reshape((bt.shape[0], 1))
        cov -= omega @ omega.T / (1 + jnp.dot(omega.squeeze(), selected_context))
        return key, (bt, cov)