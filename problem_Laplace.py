'''
    Problem 1: u(x, y) = sin(pi*x)sin(pi*y)
               f(x, y) = -2*pi**2*sin(pi*x)*sin(pi*y)   in [0, 1]
           --> dO: x(1-x)*y(1-y) = 0    
               u(x, y) = 0 in dO

'''
import numpy as np
import tensorflow as tf
from model_laplace import LaplaceZeroBoundarySolver

class Problem_1(LaplaceZeroBoundarySolver):
    def __init__(self, d = 2, inner_hidden_layers = [16, 16, 16, 16]):
        super(Problem_1, self).__init__(d = d, inner_hidden_layers = inner_hidden_layers)

    def A(self, X): # 0, or tensor
        return 0

    def f(self, X): # Must be tensor, not numpy
        return -2*np.pi**2*tf.sin(np.pi*X[:, 0])*tf.sin(np.pi*X[:, 1])

    def B(self, X): # Must be tensor, not numpy
        return X[:, 0]*(1-X[:, 0])*X[:, 1]*(1-X[:, 1])

    def exact_solution(self, X): # Must be numpy, not tensor
        return np.sin(np.pi*X[:, 0])*np.sin(np.pi*X[:, 1])


if __name__ == '__main__':
    # Define model
    hidden_layers = [256, 256] # 2 layers with 256 hidden nodes each layer
    # hidden_layers = [16, 16, 16, 16] # 4 layers with 16 hidden nodes each layer
    pb1 = Problem_1(inner_hidden_layers = hidden_layers)

    # Testing / Visualize data 
    n_points = 100
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    meshgrid = np.meshgrid(x, y)

    # Training data
    n_points = 40
    train_x = np.linspace(0, 1, n_points)
    train_y = np.linspace(0, 1, n_points)
    [trainX, trainY] = np.meshgrid(train_x, train_y)
    X = np.concatenate([trainX.reshape((-1, 1)), trainY.reshape((-1, 1))], axis=1)

    # Training
    pb1.train(X, \
            batch_size = 128, \
            epochs = 50, \
            exp_folder = 'problem_laplace', \
            draw_each_iter = True, \
            vis_each_epoches = 5, \
            meshgrid = meshgrid)
