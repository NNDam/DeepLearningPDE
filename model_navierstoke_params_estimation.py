'''
    Define Multi Layers Perceptron model 
    Author: DamDev
    Date: 06/12/2020
    Reference: Deep Learning for Partial Differential Equations CS230, Kailai Xu, Bella Shi, Shuyi Yin
'''
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from compute_differential import assert_shape, compute_delta_nd, compute_dt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from visualize import visualize_loss_error
from scipy.interpolate import griddata
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from visualize import visualize_loss_error

def prelu(_x, name = 'prelu', reuse = False):
    """
    Parametric ReLU
    """
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.1),
                        dtype=tf.float64, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - tf.abs(_x)) * 0.5

    return pos + neg

def create_mlp_model(X, hidden_layers: list, name: str, reuse = False, prelu_activation = False):
    '''
        Create MLP model from given hidden layers (number of layers & number of nodes each layer)
    '''
    with tf.variable_scope(name, reuse = reuse):
        for layer_id, layer_nodes in enumerate(hidden_layers):
            if not prelu_activation:
                X = tf.layers.dense(X, layer_nodes, activation=tf.nn.tanh, name="dense{}".format(layer_id), reuse=reuse)
            else:
                X = tf.layers.dense(X, layer_nodes, activation=None, name="dense{}".format(layer_id), reuse=reuse)
                X = prelu(X, name = 'prelu{}'.format(layer_id), reuse = reuse)
        X = tf.layers.dense(X, 2, activation=None, name="last", reuse=reuse)
    return X

def plot_solution(X_star, u_star, index, save_path = None):
    plt.clf()
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)

class NavierStokeEstimator(object):
    def __init__(self, d = 2, hidden_layers = [256]):
        # Config
        self.dimension   = d
        self.learning_rate  = tf.placeholder(tf.float64)
        self.hidden_layers = hidden_layers

        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

        # Session
        self.session = tf.Session()

        # Initializer all variables
        # self.session.run([tf.global_variables_initializer()])
        # Reuse model_Boundary & model_PDE to compute u_predict
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)

        self.loss_sumary = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred))
        # Saver
        self.saver = tf.train.Saver()
        # Otp
        self.opt_sumary = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_sumary)
        self.session.run([tf.global_variables_initializer()])

    def get_random_batch(self, X, U, batch_size = 32):
        '''
            Get index of sample for generating batch
        '''
        X, U = shuffle(X, U)
        return X[:batch_size], U[:batch_size]

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        
        psi_and_p = create_mlp_model(tf.concat([x,y,t], 1), hidden_layers = self.hidden_layers, name = 'model', reuse = False)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]  
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v

    def predict(self, x_star, y_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.session.run(self.u_pred, tf_dict)
        v_star = self.session.run(self.v_pred, tf_dict)
        p_star = self.session.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star

    def restore(self, exp_folder):
        self.saver.restore(self.session, os.path.join(exp_folder, 'model.ckpt'))

    def train_combine(self, X, U, \
            batch_size = 512, \
            steps = 1000, \
            exp_folder = 'exp', \
            vis_each_iters = 100, \
            meshgrid = None, \
            timespace = None, \
            lr_init = 0.001, \
            lr_scheduler = [4000, 6000, 8000]):
        '''
            Training combine two loss functions
        '''
        # Define sum of two loss functions & optimizer
        ## Rebuild boundary loss using normal X placeholder (old: X_boundary placeholder)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
        
        lr = lr_init
        ls_loss = []
        for it in range(steps):
            if it in lr_scheduler:
                lr = lr / 10

            batchX, batchU = self.get_random_batch(X, U, batch_size = batch_size)
            
            _, loss = self.session.run([self.opt_sumary, self.loss_sumary], \
                    feed_dict={self.x_tf: batchX[:, 1:2], \
                                self.y_tf: batchX[:, 2:3], \
                                self.t_tf: batchX[:, 0:1], \
                                self.u_tf: batchU[:, 0:1], \
                                self.v_tf: batchU[:, 1:2], \
                                self.learning_rate: lr
                                })
            
            ########## record loss ############
            ls_loss.append(loss)

            if it % 10 == 0:
                print("Iteration={}, Total Loss: {}".format(it, loss))
        self.saver.save(self.session, os.path.join(exp_folder, 'model.ckpt'))
        visualize_loss_error(ls_loss, path = os.path.join(exp_folder, 'loss.png'))



