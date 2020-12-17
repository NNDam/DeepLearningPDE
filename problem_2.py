'''
    Problem 2: u(x, y) = y**0.6
               f(x, y) = 0.6*(0.6-1)*y^(0.6-2)   in [0, 1]
           --> dO: x(1-x)*y(1-y) = 0    

'''
import numpy as np
import tensorflow as tf
from model import LaplaceBoundarySolver
from sklearn.utils import shuffle
from compute_differential import assert_shape, compute_delta_nd
from visualize import visualize_f

def func(X):
    return X[:, 1]**0.6

class Problem_2(LaplaceBoundarySolver):
    def __init__(self, d= 2, inner_hidden_layers = [256, 256, 256], boundary_hidden_layers = [256, 256, 256]):
        super(Problem_2, self).__init__(d = d, inner_hidden_layers = inner_hidden_layers, boundary_hidden_layers = boundary_hidden_layers)

    def f(self, X): # Must be tensor, not numpy
        return 0.6*(0.6-1)*X[:,1]**(0.6-2)

    def B(self, X): # Must be tensor, not numpy
        return X[:, 0]*(1-X[:, 0])*X[:, 1]*(1-X[:, 1])

    def exact_solution(self, X): # Must be numpy, not tensor
        return X[:, 1]**0.6

    def tf_exact_solution(self, X): # Must be tensor, not numpy
        return tf.pow(X[:, 1], 0.6)

if __name__ == '__main__':
    pb1 = Problem_2()

    # Visualize data 
    n_points = 100
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    meshgrid = np.meshgrid(x, y)
    visualize_f(meshgrid, func, 'solution.png')

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
            steps = 900, \
            exp_folder = 'problem2_grid', \
            vis_each_iters = 30, \
            meshgrid = meshgrid, train_method = 'COMBINE')
