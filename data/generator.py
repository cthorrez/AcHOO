import jax
import jax.numpy as jnp

def generate_linear_dataset(n_samples, n_features, sigma, concat_one=True, seed=0):
    key = jax.random.PRNGKey(seed)
    w = jax.random.normal(key, shape=(n_features+concat_one,1))

    _, key = jax.random.split(key)
    X = jax.random.normal(key, shape=(n_samples, n_features))
    if concat_one:
        X = jnp.hstack([X, jnp.ones((X.shape[0],1))])

    _, key = jax.random.split(key)
    noise = jax.random.normal(key, shape=(n_samples, 1)) * (sigma * jnp.sqrt(n_features + concat_one))
    y = jnp.dot(X, w) + noise
    return X, y

