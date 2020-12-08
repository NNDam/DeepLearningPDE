'''
    Problem 1: u(x, y) = sin(pi*x)sin(pi*y)
               f(x, y) = -2*pi**2*sin(pi*x)*sin(pi*y)   in [0, 1]
           --> dO: x(1-x)*y(1-y) = 0    
               u(x, y) = 0 in dO

'''
import numpy as np
import tensorflow as tf
from model import DeepLearningPDE
import tqdm

class Problem_1(DeepLearningPDE):
    def __init__(self, d = 2, inner_hidden_layers = [128]):
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
    pb1 = Problem_1()
    # Random data 
    n_examples = 1000
    X = np.random.rand(n_examples, 2)
    pb1.train(X, batch_size = 32, epochs = 10)
