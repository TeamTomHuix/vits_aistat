import jax
import jax.numpy as jnp

class LinUCB(object):
    def __init__(self, info, utils_object):
        self.info = info
        self.utils_object = utils_object

    def choice_fct(self, key, context, utils_vector):
        bt, cov = utils_vector
        norm = jnp.sqrt((1 / self.info.eta) * jnp.einsum("id,dv,iv->i", context, cov, context))
        data_term = jnp.matmul(context, jnp.matmul(cov, bt)).squeeze()
        p = data_term + norm
        return key, (bt, cov), p.argmax()

    def update_fct(self, key, context, action, reward, utils_vector):
        bt, cov = utils_vector
        selected_context = context[action, :]
        bt += (reward * selected_context).reshape((bt.shape[0], 1))
        omega = (cov @ selected_context).reshape((bt.shape[0], 1))
        cov -= omega @ omega.T / (1 + jnp.dot(omega.squeeze(), selected_context))
        return key, (bt, cov)

