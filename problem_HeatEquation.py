'''
    Problem Heat Equation: d_t(u) - (d_xx(u) + d_yy(u)) = (1+2pi^2)e^t.sin(pi*x)sin(pi*y)
               O = [0, 1]x[0, 1]
               t = [0, 1]
               u(0, x, y) = sin(pi*x)*sin(pi*y)
               u(., x, y)|dO = 0
           --> dO: x(1-x)*y(1-y) = 0
            u_exact = e^t.sin(pi*x)sin(pi.y)    
'''
import os
import numpy as np
import tensorflow as tf
from model_heatequation import HeatEquationSolver
from sklearn.utils import shuffle
from compute_differential import assert_shape, compute_delta_nd, compute_dt
from visualize import visualize_f

class Problem_5(HeatEquationSolver):
    def __init__(self, d= 2, hidden_layers = [128, 128, 128]):
        super(Problem_5, self).__init__(d = d, hidden_layers = hidden_layers)

    def f(self, X): # Must be tensor, not numpy
        return (1+2*np.pi*np.pi)*tf.exp(X[:, 0])*tf.sin(np.pi*X[:, 1])*tf.sin(np.pi*X[:, 2])

    def exact_solution(self, X): # Must be numpy, not tensor
        return np.exp(X[:, 0])*np.sin(np.pi*X[:, 1])*np.sin(np.pi*X[:, 2])

    def tf_exact_solution(self, X): # Must be tensor, not numpy
        return tf.exp(X[:, 0])*tf.sin(np.pi*X[:, 1])*tf.sin(np.pi*X[:, 2])

    def u0(self, X):
        return tf.sin(np.pi*X[:, 1])*tf.sin(np.pi*X[:, 2])

if __name__ == '__main__':
    # Define model
    hidden_layers = [128, 128, 128]
    pb5 = Problem_5(hidden_layers = hidden_layers)

    # Testing / Visualize data 
    n_points = 100
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    test_t = np.linspace(0, 1.0, 3)
    meshgrid = np.meshgrid(x, y)

    # Training data
    n_points_t = 4 # 0, 0.25, 0.5, 1.0
    n_points_inner = 60 # 60x60xT
    n_points_boundary = 128 # 128x4
    # Template <x, 0>
    tmp00   = np.random.rand(n_points_boundary, 1)
    tmp01   = np.zeros((n_points_boundary, 1))
    tmp0    = np.concatenate([tmp00, tmp01], axis = 1)
    # Template <x, 1>
    tmp10   = np.random.rand(n_points_boundary, 1)
    tmp11   = np.ones((n_points_boundary, 1))
    tmp1    = np.concatenate([tmp10, tmp11], axis = 1)
    # Template <0, y>
    tmp20   = np.zeros((n_points_boundary, 1))
    tmp21   = np.random.rand(n_points_boundary, 1)
    tmp2    = np.concatenate([tmp20, tmp21], axis = 1)
    # Template <1, y>
    tmp30   = np.ones((n_points_boundary, 1))
    tmp31   = np.random.rand(n_points_boundary, 1)
    tmp3    = np.concatenate([tmp30, tmp31], axis = 1)
    # Concatenate
    X_bound = np.concatenate([tmp0, tmp1, tmp2, tmp3], axis = 0)
    t_bound = np.random.rand(len(X_bound), 1)
    train_x = np.linspace(0, 1, n_points_inner)
    train_y = np.linspace(0, 1, n_points_inner)
    [trainX, trainY] = np.meshgrid(train_x, train_y)
    X = np.concatenate([trainX.reshape((-1, 1)), trainY.reshape((-1, 1))], axis=1)
    t = np.linspace(0, 1.0, n_points_t)
    X_tile = np.tile(X, (n_points_t, 1))
    t_tile = np.repeat(t, len(X), 0).reshape((-1, 1))

    # Training
    pb5.train_combine(np.concatenate([t_tile, X_tile], axis = 1), np.concatenate([t_bound, X_bound], axis = 1), \
            batch_size = 128*4, \
            steps = 10000, \
            exp_folder = 'problem_heatequation', \
            vis_each_iters = 100, \
            meshgrid = meshgrid, \
            timespace = test_t, \
            lr_init = 1e-3, \
            lr_scheduler = [4000, 6000, 8000]) # Recommend using SEPARATE method
