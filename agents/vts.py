import jax
import jax.numpy as jnp

class VTS(object):
    def __init__(self, info, utils_object):
        self.info = info
        self.utils_object = utils_object
        self.alpha_0 = self.utils_object.alpha_0
        self.beta_0 = self.utils_object.beta_0
        self.gamma_0 = self.utils_object.gamma_0
        self.theta_0 = self.utils_object.theta_0
        self.Sigma_0 = self.utils_object.Sigma_0
        self.Sigma_0_inv = self.utils_object.Sigma_0_inv

    def sample_posterior(self, key, alpha, beta, Sigma, theta):
        def sample_inverse_gamma(key, alpha, beta):
            key, subkey = jax.random.split(key)
            return key, beta / (jax.random.gamma(subkey, alpha))

        def sample_multivariate_gaussian(tree):
            theta, covariance_matrix, subkey = tree
            return jax.random.multivariate_normal(subkey, theta, covariance_matrix)

        key, sigma_sample = sample_inverse_gamma(key, alpha, beta)
        cov = jnp.multiply(sigma_sample.reshape((self.info.nb_arms, self.info.vts.nb_mixtures, 1, 1)), Sigma)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self.info.nb_arms * self.info.vts.nb_mixtures).reshape((self.info.nb_arms, self.info.vts.nb_mixtures, -1))
        sample = jax.vmap(jax.vmap(sample_multivariate_gaussian, in_axes=((0, 0, 0),)),in_axes=((0, 0, 0),))((theta, cov, keys))
        return key, sample

    def choice_fct(self, key, context, utils_vector):
        features, actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv = utils_vector
        key, theta_sample = self.sample_posterior(key, alpha, beta, Sigma, theta)
        rewards_per_mixture = jnp.einsum("ad, akd-> ak", context, theta_sample)
        pi_matrix = gamma / gamma.sum(1, keepdims=True)
        reward_per_action = jnp.einsum("ak, ak -> a", pi_matrix, rewards_per_mixture)
        utils_vector = features, actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv,
        return key, utils_vector, reward_per_action.argmax()
    
    def update_law(self, idx, features, actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv):
        def get_r(a, b, g, g_sum, t, S):
            t1 = -0.5 * (jnp.log(b) - jax.scipy.special.digamma(a))
            t2 = -0.5 * jnp.einsum("id, dv, iv-> i", features, S, features)
            t3 = -0.5 * jnp.square(labels - jnp.matmul(features, t)) * a / b
            t4 = jax.scipy.special.digamma(g) - jax.scipy.special.digamma(g_sum)
            log_r = t1 + t2 + t3 + t4
            return jnp.exp(log_r - jnp.max(log_r))

        def update_parameters(r, a0, b0, g0, t0, s_inv0, ca):
            R = jnp.diag(r * jnp.where(ca, 1, 0))
            g_new = g0 + jnp.trace(R)
            s_inv_new = jnp.einsum("td,tv,ve->de", features, R, features) + s_inv0
            s_new = jnp.linalg.pinv(s_inv_new)
            t_new = jnp.matmul(s_new, jnp.einsum("td,tv,v->d", features, R, labels) + jnp.matmul(s_inv0, t0))
            a_new = a0 + 0.5 * jnp.trace(R)
            b_new = b0 + 0.5 * (jnp.matmul(jnp.matmul(labels, R), labels)) + 0.5 * (jnp.matmul(jnp.matmul(t0, s_inv0), t0) - jnp.matmul(jnp.matmul(t_new, s_inv_new), t_new))
            return a_new, b_new, g_new, t_new, s_new, s_inv_new

        gamma_sum = jnp.sum(gamma, axis=1, keepdims=True).repeat(self.info.vts.nb_mixtures, axis=1)
        r = jax.vmap(jax.vmap(get_r))(alpha, beta, gamma, gamma_sum, theta, Sigma)

        r /= jnp.sum(r, axis=1, keepdims=True)
        chosen_action = actions == jnp.tile(jnp.expand_dims(jnp.arange(self.info.nb_arms), axis=(1, 2)), (1, self.info.vts.nb_mixtures, actions.shape[0]))
        alpha, beta, gamma, theta, Sigma, Sigma_inv = jax.vmap(jax.vmap(update_parameters))(r,
                                                                                            self.alpha_0,
                                                                                            self.beta_0,
                                                                                            self.gamma_0,
                                                                                            self.theta_0,
                                                                                            self.Sigma_0_inv,
                                                                                            chosen_action)
        return alpha, beta, gamma, theta, Sigma, Sigma_inv


    def update_fct(self, key, context, action, reward, utils_vector):
        features, actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv = utils_vector
        feature = context[action]
        features = features.at[-1].set(feature)
        labels = labels.at[-1].set(reward)
        actions = actions.at[-1].set(action)

        (alpha, beta, gamma, theta, Sigma, Sigma_inv) =jax.lax.fori_loop(
             0,
             self.info.vts.num_updates,
             lambda i, x: self.update_law(i, features, actions, labels, *x),
        (alpha, beta, gamma, theta, Sigma, Sigma_inv))

        return key, (features, actions, labels, alpha, beta, gamma, theta, Sigma, Sigma_inv)