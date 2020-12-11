'''
    Problem 4 (Peak): u(x, y) = exp(-1000((x-0.5)**2 + (y-0.5)**2))
               f(x, y) = -4*1000*u(x, y) + 4*1000**2*u(x, y)*((x-0.5)**2 + (y-0.5)**2) on (0, 1)
           --> dO: x(1-x)*y(1-y) = 0    
'''
import numpy as np
import tensorflow as tf
from model import LaplaceBoundarySolver
from sklearn.utils import shuffle
from compute_differential import assert_shape, compute_delta_nd
from visualize import visualize_f

def func(X):
    return np.exp(-1000*((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2))

class Problem_2(LaplaceBoundarySolver):
    def __init__(self, d= 2, inner_hidden_layers = [256, 256], boundary_hidden_layers = [256, 256]):
        super(Problem_2, self).__init__(d = d, inner_hidden_layers = inner_hidden_layers, boundary_hidden_layers = boundary_hidden_layers)

    def f(self, X): # Must be tensor, not numpy
        return -4*1000*self.tf_exact_solution(X) + 4*1000**2*self.tf_exact_solution(X)*(tf.pow(X[:, 0] - 0.5, 2) + tf.pow(X[:, 1] - 0.5, 2))

    def B(self, X): # Must be tensor, not numpy
        return X[:, 0]*(1-X[:, 0])*X[:, 1]*(1-X[:, 1])

    def exact_solution(self, X): # Must be numpy, not tensor
        return np.exp(-1000*((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2))

    def tf_exact_solution(self, X): # Must be tensor, not numpy
        return tf.exp(-1000*((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2))

if __name__ == '__main__':
    import os
    pb1 = Problem_2()

    # Visualize data 
    n_points = 100
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    meshgrid = np.meshgrid(x, y)
    visualize_f(meshgrid, func, os.path.join('problem4_grid', 'solution.png'))

    # Random data 
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
    X_bound = shuffle(X_bound)
    # pb1.train(X, X_bound, \
    #         batch_size = 16, \
    #         steps = 300, \
    #         exp_folder = 'problem2', \
    #         vis_each_iters = 30, \
    #         meshgrid = meshgrid)
    # Meshdata (train = subset(test))
    n_points = 40
    train_x = np.linspace(0, 1, n_points)
    train_y = np.linspace(0, 1, n_points)
    [trainX, trainY] = np.meshgrid(x, y)
    X = np.concatenate([trainX.reshape((-1, 1)), trainY.reshape((-1, 1))], axis=1)
    pb1.train(X, X_bound, \
            batch_size = 16, \
            steps = 3000, \
            exp_folder = 'problem4_grid', \
            vis_each_iters = 30, \
            meshgrid = meshgrid)
