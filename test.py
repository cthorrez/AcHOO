from functools import partial
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

from jax.config import config
config.update("jax_enable_x64", True)

import data.loader
import data.generator
from optimizers import GradientDescent, AcHOO

@jit
def mse(preds, labels):
    return jnp.mean(jnp.square(preds - labels))

@jit
def mpe(preds, labels, power):
    return jnp.power(jnp.mean(jnp.power(preds - labels, power)),1/1)

@jit
def cross_entropy(logits, labels):
    """preds is [batch,n_classes], labels is [batch_size,]"""
    return -1 * ((logits - jax.scipy.special.logsumexp(logits, axis=1)[:,None]) * labels).mean()

@jit
def binary_cross_entropy(logits, labels):
    max_val = jnp.clip(logits, 0, None)
    loss = logits - logits * labels + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-logits - max_val)))
    return loss.mean()

@jit
def softmax(x):
    exp = jnp.exp(x)
    return exp / jnp.sum(exp, axis=1)[:,None]

@jit
def linear(w, X):
    return jnp.dot(X, w)

def linear_regression_loss(w, X, y, reg, power=2):
    preds = linear(w,X)
    return mpe(preds, y, power) + (reg * jnp.power(w[:-1], 2).sum())

@jit
def logistic_regression_prediction(w, X):
    logits = linear(w, X)
    return softmax(logits)

@jit
def logistic_regression_loss(w, X, y, reg):
    logits = linear(w, X)
    ce_loss = cross_entropy(logits, y)
    return ce_loss + (reg * jnp.power(w[:,:-1], 2).sum())

@jit
def binary_logistic_regression_loss(w, X, y, reg):
    logits = linear(w, X)
    ce_loss = binary_cross_entropy(logits, y)
    return ce_loss + (reg * jnp.power(w[:,:-1], 2).sum())



def one_hot(labels, n_classes):
    return labels[:, None] == jnp.arange(n_classes)

def linear_regression():

    X, y = data.generator.generate_linear_dataset(
        n_samples=10,
        n_features=1,
        sigma=0.2,
        concat_one=False
    )
    n_features = X.shape[1]
    reg = 0.01
    loss_power = 8
    w = jnp.zeros(n_features)
    print(f'fitting {w.size} parameters')
    opt_fn = partial(linear_regression_loss, X=X, y=y, reg=reg, power=loss_power)
    run_opt(opt_fn, w)

def run_opt(opt_fn, w):

    gd = GradientDescent(
        fn=opt_fn,
        params=w,
        max_iter=10000
    )
    w_gd, values_gd = gd.fit()
    plt.plot(
        jnp.arange(values_gd.shape[0]),
        jnp.log(values_gd) / jnp.log(100),
        label='Gradient Descent',
        marker='o',
        markersize=8,
        color='blue',
        alpha=0.5,
    )

    orders = [2,3,4,7]
    colors = ['red', 'green', 'orange', 'purple', 'black']
    markers = ['X', 'v', '*', '^']

    for i, order in enumerate(orders):
        if order == 2 : label = "2nd Order (Newton's Method)"
        elif order == 3 : label = "3rd Order (Halley's Method)"
        else: label = f'{order}th Order'

        achoo = AcHOO(fn=opt_fn, 
                        params=w,
                        order=order, 
                        step_size=1.0,
                        max_iter=25)

        w_achoo, values_achoo = achoo.fit()
        plt.plot(
            jnp.arange(values_achoo.shape[0]),
            jnp.log(values_achoo) / jnp.log(100),
            label=label,
            marker=markers[i],
            markersize=8,
            color=colors[i],
            alpha=0.5,
        )

    plt.legend()
    plt.show()



def logistic_regression():
    key = jax.random.PRNGKey(0)
    X, y = data.loader.load_iris()

    y = y.at[y==1].set(0)

    n_features = X.shape[1]
    n_classes = jnp.unique(y).shape[0]
    n_params = n_features * n_classes
    reg = 0.1
 
    w = jax.random.normal(key, shape=(n_features, n_classes)) * 0.01
    # w = jnp.array([[-0.15478352,0.21257754,-0.40969526],[1.54926772,0.65786876,1.17450817],[-0.49462997,0.54060939,-0.04389029],[-0.02852532,-2.22300611,0.74312746],[0.9482618,0.45170073,0.81152022]])
    w = jnp.zeros((n_features, n_classes))
    opt_fn = partial(logistic_regression_loss, X=X, y=one_hot(y, n_classes), reg=reg)
    
    w = jnp.zeros((n_features, 1))
    opt_fn = partial(binary_logistic_regression_loss, X=X, y=y, reg=reg)

    print(f'fitting on {w.size} parameters')
    run_opt(opt_fn, w)



if __name__ == '__main__':
    # main()
    linear_regression()
    # logistic_regression()
