import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev


@jit
def get_direction(grads):
    directions = [-1.0 * grads[1], -1.0 * jnp.dot(jnp.linalg.inv(grads[2]), grads[1])]
    denom = 1.0
    sum_terms = [grads[2]]
    prev_direction = 1.0
    for i, grad in enumerate(grads[3:], 2):
        denom = denom * i
        intermediates = [grad / denom]
        for j in range(1, len(directions)):
            intermediates.append(jnp.dot(intermediates[-1], directions[-j]))
        sum_terms.append(intermediates[-1])
        new_direction = jnp.dot(jnp.linalg.inv(jnp.sum(jnp.stack(sum_terms), axis=0)), directions[0])
        directions.append(new_direction)
    return directions[-1]

class AcHOO:
    def __init__(self, fn, params, order, step_size, tol=1e-6, max_iter=100):
        self.fn = fn
        self.params = params
        self.n_params = jnp.size(params)
        self.order = order
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter

        self.grad_functions = [fn]
        for i in range(order):
            if i <= 1 :
                self.grad_functions.append(jacrev(self.grad_functions[-1]))
            else:
                self.grad_functions.append(jacfwd(self.grad_functions[-1]))
    

    def fit(self):

        @jit
        def get_grads(params):
            """helper function to get a list of gradients, grads[i] contains the ith degree gradient 
            (0th degree is the function value itself, 1st is gradient, 2nd is Hessian, etc)"""
            grads = []
            for i, grad_fn in enumerate(self.grad_functions):
                grads.append(grad_fn(params).reshape((self.n_params,) * i))
            return grads

        values = [jnp.inf, jnp.inf]
        for i in range(self.max_iter):
            grads = get_grads(self.params)
            print(f'iteration {i} loss: {grads[0]}')

            if jnp.isinf(grads[0]) or jnp.isnan(grads[0]):
                print(f'optimization diverged at iteration {i}')
                break

            if (jnp.abs(grads[0].item() - values[-1]) < self.tol) or (jnp.abs(grads[0].item() - values[-2]) < self.tol):
                print(f'converged in {i} iterations')
                break

            if self.order == 1:
                direction = -grads[1]
            else:
                direction = get_direction(grads)

            self.params = self.params + self.step_size * direction.reshape(self.params.shape)
            prev_value = grads[0]
            values.append(grads[0].item())

        return self.params, jnp.array(values[2:])


class GradientDescent:
    def __init__(self, fn, params, alpha=0.5, beta=0.5, tol=1e-6, max_iter=100):
        self.fn = jit(fn)
        self.grad_fn = jit(jax.jacrev(fn))
        self.params = params
        self.n_params = jnp.size(params)
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter

    def fit(self):

        values = [jnp.inf, jnp.inf]
        for i in range(self.max_iter):
            f_x = self.fn(self.params)
            print(f'iteration {i} loss: {f_x}')

            if jnp.isinf(f_x) or jnp.isnan(f_x):
                print(f'optimization diverged at iteration {i}')
                break

            if (jnp.abs(f_x.item() - values[-1]) < self.tol) or (jnp.abs(f_x.item() - values[-2]) < self.tol):
                print(f'converged in {i} iterations')
                break

            g = self.grad_fn(self.params)
            norm = jnp.dot(g.flatten(), -g.flatten())
            t = 1.0
            while self.fn(self.params - t*g) > f_x + (self.alpha * t * norm):
                t = t * self.beta

            self.params = self.params - (t * g.reshape(self.params.shape))
            values.append(f_x.item())

        return self.params, jnp.array(values[2:])

        







