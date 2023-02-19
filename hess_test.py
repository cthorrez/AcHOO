import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
from solution import BinaryLogisticRegression
import numpy
from jax.config import config
# config.update("jax_enable_x64", True)




def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))

def get_probs(w, X):
    logits = np.dot(X, w)
    return sigmoid(logits)



def objective(w, X, y, lmbda):
    p = get_probs(w, X)
    first_term = y * np.log(p)
    second_term = (1. - y) * (np.log(1. - p ))
    L = -1. * np.sum(first_term + second_term, axis=0)
    L += 0.5 * lmbda * np.dot(w,w)
    return L



if __name__ == '__main__':
    
    lmbda = 0.001
    model = BinaryLogisticRegression(lmbda, None)

    data = numpy.loadtxt('solution_files/data_banknote_authentication.txt', 
                    delimiter=',', dtype=np.float32)
    data = np.array(data)
    X = data[:,:-1]
    y = data[:,-1]
    n,d = X.shape
    d += 1
    key = random.PRNGKey(0)
    w = random.uniform(key, (d,))
    X = np.hstack([X, np.ones((n,1), dtype=np.float32)])


    key = random.PRNGKey(0)
    alpha = 0.001

    hessian = lambda f : jacrev(jacrev(f))
    grad = lambda f : jacfwd(f)
    super_hessian = lambda f : jacfwd(jacfwd(jacfwd(jacrev(jacrev(f)))))

    
    def mega_hess(f, n=20):
        x = f
        for i in range(n):
            x = jacfwd(x)
        return x



    # g_my = model.gradient(w, X, y)
    # g_jax = grad(objective)(w, X, y, lmbda)


    # H_my = model.hessian(w, X, y)
    # H_jax = hessian(objective)(w, X, y, lmbda)
    # H_jax = numpy.array(H_jax)


    # print(H_my)
    # print(H_jax)

    # print(np.max(np.abs(H_jax - H_my)))


    # print(g_my)
    # print(g_jax)
    # print(np.max(np.abs(g_jax - g_my)))

    # sH = super_hessian(objective)(w, X, y, lmbda)
    # print(sH.shape)
    # print(sH)


    X = X[:,0:2]
    w = w[0:2]

    # X = X[0,:]
    # y = y[:0]

    mH = mega_hess(objective, 12)(w, X, y, lmbda)
    print(mH.shape)