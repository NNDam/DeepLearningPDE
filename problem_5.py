'''
    Problem 5 (Heat Equation): d_t(u) - (d_xx(u) + d_yy(u)) = (1+2pi^2)e^t.sin(pi*x)sin(pi*y)
               O = [0, 1]x[0, 1]
               t = [0, 1]
               u(0, x, y) = sin(pi*x)*sin(pi*y)
               u(., x, y)|dO = 0
           --> dO: x(1-x)*y(1-y) = 0
            u_exact = e^t.sin(pi*x)sin(pi.y)    
'''
import numpy as np
import tensorflow as tf
from model_heatequation import HeatEquationSolver
from sklearn.utils import shuffle
from compute_differential import assert_shape, compute_delta_nd, compute_dt
from visualize import visualize_f

def func(X):
    return np.exp(-1000*((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2))

class Problem_2(HeatEquationSolver):
    def __init__(self, d= 2, inner_hidden_layers = [10, 10, 10, 10], boundary_hidden_layers = [10, 10, 10, 10]):
        super(Problem_2, self).__init__(d = d, inner_hidden_layers = inner_hidden_layers, boundary_hidden_layers = boundary_hidden_layers)

    def f(self, X): # Must be tensor, not numpy
        return (1+2*np.pi**2)*tf.exp(X[:, 0])*tf.sin(np.pi*X[:, 1])*tf.sin(np.pi*X[:, 2])

    def B(self, X): # Must be tensor, not numpy
        return X[:, 1]*(1-X[:, 1])*X[:, 2]*(1-X[:, 2])

    def exact_solution(self, X): # Must be numpy, not tensor
        return np.exp(X[:, 0])*np.sin(np.pi*X[:, 1])*np.sin(np.pi*X[:, 2])

    def tf_exact_solution(self, X): # Must be tensor, not numpy
        return tf.exp(X[:, 0])*tf.sin(np.pi*X[:, 1])*tf.sin(np.pi*X[:, 2])

if __name__ == '__main__':
    import os
    pb1 = Problem_2()

    # Visualize data 
    n_points = 50
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    meshgrid = np.meshgrid(x, y)
    timespace = np.linspace(0, 1, 2)
    # visualize_f(meshgrid, func, os.path.join('problem5_grid', 'solution.png'))

    # Training data
    n_examples = 500
    X      = np.random.rand(n_examples, 2)
    # Template <x, 0>
    tmp00   = np.random.rand(100, 1)
    tmp01   = np.zeros((100, 1))
    tmp0    = np.concatenate([tmp00, tmp01], axis = 1)
    # Template <x, 1>
    tmp10   = np.random.rand(100, 1)
    tmp11   = np.ones((100, 1))
    tmp1    = np.concatenate([tmp10, tmp11], axis = 1)
    # Template <0, y>
    tmp20   = np.zeros((100, 1))
    tmp21   = np.random.rand(100, 1)
    tmp2    = np.concatenate([tmp20, tmp21], axis = 1)
    # Template <1, y>
    tmp30   = np.ones((100, 1))
    tmp31   = np.random.rand(100, 1)
    tmp3    = np.concatenate([tmp30, tmp31], axis = 1)
    # Concatenate
    X_bound = np.concatenate([tmp0, tmp1, tmp2, tmp3], axis = 0)
    t_bound = np.random.rand(len(X_bound), 1)
    n_points = 40
    train_x = np.linspace(0, 1, n_points)
    train_y = np.linspace(0, 1, n_points)
    [trainX, trainY] = np.meshgrid(x, y)
    X = np.concatenate([trainX.reshape((-1, 1)), trainY.reshape((-1, 1))], axis=1)
    t = np.random.rand(len(X), 1)
    pb1.train(np.concatenate([t, X], axis = 1), np.concatenate([t_bound, X_bound], axis = 1), \
            batch_size = 16, \
            steps = 10000, \
            exp_folder = 'problem5_grid', \
            vis_each_iters = 100, \
            meshgrid = meshgrid, \
            train_method = 'COMBINE', \
            timespace = timespace, \
            lr_init = 0.01, \
            lr_scheduler = [4000, 6000, 8000]) # Recommend using SEPARATE method
