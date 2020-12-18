'''
    Problem 6 (Steady Navier Stoke): -v*div(grad(u)) + u.grad(u) + grad(p) = f
               div(u) = 0 in O
               u|dO = g
               O = [-0.5, 1.0]x[-0.5, 1.5]
               t = [0, 1]
               u = (u1, u2)
               v = 0.025
               lamb = 1/(2v) - sqrt(1/(4v^2) + 4pi^2)
               u1(x1, x2) = 1-exp(lamb*x1)cos(2*pi*x2)
               u2(x1, x2) = lambda/2pi*exp(lamb*x1)cos(2*pi*x2)
               p(x1, x2)  = 1/2(1-exp(2*lamb*x1)) + C
               f = 0
           
'''
import numpy as np
import tensorflow as tf
from model_navierstoke import NavierStokeSolver
from sklearn.utils import shuffle
from compute_differential import assert_shape, compute_delta_nd, compute_dt
from visualize import visualize_f

class Problem_6(NavierStokeSolver):
    def __init__(self, d= 2, velocity_hidden_layers = [10, 10, 10, 10], pressure_hidden_layers = [10, 10, 10, 10], batch_size = 32):
        super(Problem_6, self).__init__(d = d, velocity_hidden_layers = velocity_hidden_layers, pressure_hidden_layers = pressure_hidden_layers, batch_size = batch_size)
        self.v = 0.025
        self.lamb = 1/(2*self.v) - np.sqrt(1/(4*self.v*self.v) + 4*np.pi*np.pi)

    def f(self, X): # Must be numpy
        '''
            Input: X = [x1, x2]
            Return: f(X)
        '''
        return np.zeros(X.shape)

    def u1(self, X): # Must be numpy
        return 1 - np.exp(self.lamb*X[:, 0])*np.cos(2*np.pi*X[:, 1])

    def u2(self, X): # Must be numpy
        return self.lamb/(2*np.pi)*np.exp(self.lamb*X[:, 0])*np.sin(2*np.pi*X[:, 1])

    def p(self, X): # Must be numpy
        return 0.5*(1-np.exp(2*self.lamb*X[:, 0]))

    def exact_solution(self, X): #
        return np.concatenate([self.u1(X).reshape((-1, 1)), self.u2(X).reshape((-1, 1))], axis = 1)

if __name__ == '__main__':
    import os
    pb1 = Problem_6()

    # Visualize data 
    n_points = 50
    x = np.linspace(-0.5, 1.0, n_points)
    y = np.linspace(-0.5, 1.5, n_points)
    vis_meshgrid = np.meshgrid(x, y)
    # visualize_f(meshgrid, func, os.path.join('problem5_grid', 'solution.png'))

    # Training data
    # Template <x, -0.5>
    tmp00   = np.random.rand(100, 1) * (1.0 - -0.5) + (-0.5)
    tmp01   = np.ones((100, 1))*-0.5
    tmp0    = np.concatenate([tmp00, tmp01], axis = 1)
    # Template <x, 1.5>
    tmp10   = np.random.rand(100, 1) * (1.0 - -0.5) + (-0.5)
    tmp11   = np.ones((100, 1))*1.5
    tmp1    = np.concatenate([tmp10, tmp11], axis = 1)
    # Template <-0.5, y>
    tmp20   = np.ones((100, 1))*(-0.5)
    tmp21   = np.random.rand(100, 1) * (1.5 - -0.5) + (-0.5)
    tmp2    = np.concatenate([tmp20, tmp21], axis = 1)
    # Template <1.0, y>
    tmp30   = np.ones((100, 1))
    tmp31   = np.random.rand(100, 1) * (1.5 - -0.5) + (-0.5)
    tmp3    = np.concatenate([tmp30, tmp31], axis = 1)
    # Concatenate
    X_bound = np.concatenate([tmp0, tmp1, tmp2, tmp3], axis = 0)
    t_bound = np.random.rand(len(X_bound), 1)
    n_points = 40
    x = np.linspace(-0.5, 1.0, n_points)
    y = np.linspace(-0.5, 1.5, n_points)
    [trainX, trainY] = np.meshgrid(x, y)
    X = np.concatenate([trainX.reshape((-1, 1)), trainY.reshape((-1, 1))], axis=1)
    pb1.train_combine(X, X_bound, \
            steps = 10000, \
            exp_folder = 'problem6_grid', \
            vis_each_iters = 100, \
            meshgrid = vis_meshgrid) # Recommend using SEPARATE method
