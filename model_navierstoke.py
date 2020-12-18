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
from model import create_mlp_model

def create_mlp_model(X, out_shape, hidden_layers: list, name: str, reuse = False):
    '''
        Create MLP model from given hidden layers (number of layers & number of nodes each layer)
    '''
    with tf.variable_scope(name):
        for layer_id, layer_nodes in enumerate(hidden_layers):
            X = tf.layers.dense(X, layer_nodes, activation=tf.nn.tanh, name="dense{}".format(layer_id), reuse=reuse)
        X = tf.layers.dense(X, out_shape, activation=None, name="last", reuse=reuse)
        # X = tf.squeeze(X, axis=1)
        # assert_shape(X, (None,))
    return X

class NavierStokeSolver(object):
    '''
        Template of DeepLearning PDE models with specific inner & boundary model
    '''
    def __init__(self, d = 2, velocity_hidden_layers = [10, 10, 10, 10], pressure_hidden_layers = [10, 10, 10, 10], batch_size = 32):
        '''
            Init template with default config
            -> solver u:= Boundary(x, y, w1) + B(x, y).PDE(x, y, w2) 
                            Boundary: boundary deep learning model (Boundary = g_D in dD)
                            PDE: PDE deep learning model
        '''
        # Config
        nu = 0.025
        self.dimension   = d
        self.batch_size = batch_size
        self.X           = tf.placeholder(tf.float64, (None, self.dimension))
        self.X_boundary  = tf.placeholder(tf.float64, (None, self.dimension))
        self.X_press     = tf.placeholder(tf.float64, (None, self.dimension))
        self.learning_rate = tf.placeholder(tf.float64)

        self.velocity_hidden_layers = velocity_hidden_layers
        self.pressure_hidden_layers = pressure_hidden_layers

        # Multi-layers perceptron model with tanh activation function
        self.model_velocity_int = create_mlp_model(self.X, out_shape = 2, hidden_layers = velocity_hidden_layers, name = 'velocity')
        self.model_velocity_bou = create_mlp_model(self.X_boundary, out_shape = 2, hidden_layers = velocity_hidden_layers, name = 'velocity', reuse = True)
        self.model_pressure_int = create_mlp_model(self.X_press, out_shape = 1, hidden_layers = pressure_hidden_layers, name = 'pressure')
        # self.model_PDE      = create_mlp_model(self.X, hidden_layers = inner_hidden_layers, name = 'inner')
    
        # Gradient
        grad      = self.first_derivatives_nn_velocity(self.X, self.batch_size)
        grad_grad = self.second_derivatives_nn_velocity(self.X, self.batch_size)
        grad_p    = self.first_derivates_nn_pressure(self.X_press, self.batch_size)
        p_x = tf.slice(grad_p[0][0], [0,0], [batch_size,1])
        p_y = tf.slice(grad_p[0][0], [0,1], [batch_size,1])

        self.sol_int_x = tf.placeholder(tf.float64, [None, 1])
        self.sol_int_y = tf.placeholder(tf.float64, [None, 1])

        self.sol_bou_x = tf.placeholder(tf.float64, [None, 1])
        self.sol_bou_y = tf.placeholder(tf.float64, [None, 1])

        u_xx = tf.slice(grad_grad[0][0][0], [0, 0], [batch_size, 1])
        u_yy = tf.slice(grad_grad[0][1][0], [0, 1], [batch_size, 1])
        v_xx = tf.slice(grad_grad[1][0][0], [0, 0], [batch_size, 1])
        v_yy = tf.slice(grad_grad[1][1][0], [0, 1], [batch_size, 1])

        u_x = tf.slice(grad[0][0], [0, 0], [batch_size, 1])
        u_y = tf.slice(grad[0][0], [0, 1], [batch_size, 1])

        v_x = tf.slice(grad[1][0], [0, 0], [batch_size, 1])
        v_y = tf.slice(grad[1][0], [0, 1], [batch_size, 1])

        vel_bou_x = tf.slice(self.model_velocity_bou, [0, 0], [batch_size, 1])
        vel_bou_y = tf.slice(self.model_velocity_bou, [0, 1], [batch_size, 1])

        vel_x = tf.slice(self.model_velocity_int, [0, 0], [batch_size, 1])
        vel_y = tf.slice(self.model_velocity_int, [0, 1], [batch_size, 1])

        advection_x = u_x*vel_x + u_y*vel_y
        advection_y = v_x*vel_x + v_y*vel_y

        # Loss
        self.loss_int = tf.square(-advection_x+nu*(u_xx+u_yy)+self.sol_int_x-p_x) + tf.square(-advection_y+nu*(v_xx+v_yy)+self.sol_int_y-p_y)
        self.loss_div = tf.square(u_x+v_y)
        loss_bou_x = tf.square(vel_bou_x-self.sol_bou_x)
        loss_bou_y = tf.square(vel_bou_y-self.sol_bou_y)
        self.loss_bou = loss_bou_x + loss_bou_y
        self.loss = self.loss_int + self.loss_bou + self.loss_div
        
        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        # Session
        self.session = tf.Session()

        # Saver
        self.saver = tf.train.Saver()

        # Initializer all variables
        self.session.run([tf.global_variables_initializer()])
    
    def first_derivatives_nn_velocity(self, X, batch_size):
        vec = create_mlp_model(X, out_shape = 2, hidden_layers = self.velocity_hidden_layers, name = 'velocity', reuse = True)
        grad_velocity = []
        for i in range(2):
            grad_velocity.append(tf.gradients(tf.slice(vec, [0,i], [batch_size,1]), X))
        return grad_velocity

    def second_derivatives_nn_velocity(self, X, batch_size):
        grad_velocity = self.first_derivatives_nn_velocity(X, batch_size)
        grad_grad_velocity = []
        for i in range(len(grad_velocity)):
            second_derivatives = []
            for j in range(2):
                second_derivatives.append(tf.gradients(tf.slice(grad_velocity[i][0], [0, j], [batch_size,1]), X))
            grad_grad_velocity.append(second_derivatives)
        return grad_grad_velocity

    def first_derivates_nn_pressure(self, X, batch_size):
        vec = create_mlp_model(X, out_shape = 1, hidden_layers = self.pressure_hidden_layers, name = 'pressure', reuse = True)
        batch_size = tf.shape(X)[0]
        grad_pressure = []
        for i in range(2):
            grad_pressure.append(tf.gradients(tf.slice(vec, [0,i], [batch_size,1]), X))

        return grad_pressure

    def u1(self, X):
        raise NotImplementedError

    def u2(self, X):
        raise NotImplementedError

    def f(self, X):
        raise NotImplementedError

    def p(self, X):
        raise NotImplementedError

    def get_batch(self, X, X_bound, idx, batch_size = 32):
        '''
            Get index of sample for generating batch
        '''
        X_bound = shuffle(X_bound)
        total_samples = len(X)
        is_end = False
        flag_start = idx*batch_size
        flag_end   = (idx+1)*batch_size
        if flag_end >= total_samples:
            flag_end = total_samples
            flag_start = total_samples - batch_size
            is_end = True
        # Calculate
        batch     = X[flag_start: flag_end]
        batch_bou = X_bound[:batch_size]
        return batch, \
                batch_bou, \
                self.f(batch), \
                self.u1(batch_bou), \
                self.u2(batch_bou), \
                is_end

    def get_random_batch(self, X, batch_size = 32):
        '''
            Get index of sample for generating batch
        '''
        temp = shuffle(X)
        return temp[:batch_size]

    def visualize_surface(self, meshgrid, save_path, show_only_sol = False):
        '''
            Visualize 3D surface
        '''
        print('<!> Visualize surface with meshgrid')
        X, Y = meshgrid

        vis_points = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)

        V = self.session.run(self.model_velocity_int, feed_dict={self.X: vis_points})
        V = V.reshape((len(Y), len(X), 2))
        vis_points = vis_points.reshape((len(Y), len(X), 2))
        # u_true = self.exact_solution(vis_points)

        # plt.clf()
        # fig = plt.figure()
        # plt.quiver(*vis_points.T, V[:,0], V[:,1], color=['r','b','g'], scale=21)
        # # plt.show()
        # fig.savefig(save_path)

        plt.clf()
        fig = plt.figure()
        # Varying color along a streamline
        strm = plt.streamplot(vis_points[:, :, 0], vis_points[:, :, 1], V[:, :, 0], V[:, :, 1], color=V[:, :, 0], linewidth=2, cmap='autumn')
        fig.colorbar(strm.lines)
        fig.savefig(save_path.split('.')[0] + '_streamplot.png')        
        
    def train_combine(self, X, X_boundary, \
            steps = 1000, \
            exp_folder = 'exp', \
            vis_each_iters = 100, \
            meshgrid = None, \
            timespace = None, \
            lr_init = 0.01, \
            lr_scheduler = [4000, 6000, 8000]):
        '''
            Training combine 3 loss functions
        '''
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
        # Visualize
        meshX, meshY = meshgrid
        vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
        # Training
        ls_bou       = []
        ls_int       = []
        ls_div       = []
        ls_l2        = []
        ls_total     = []
        bbatch_index = 0
        lr = lr_init
        for it in range(steps):
            if it in lr_scheduler:
                lr = lr / 10
            _batch, _batch_bou, _fxy, _u1, _u2, is_end = self.get_batch(X, X_boundary, bbatch_index, batch_size = self.batch_size)
            bbatch_index += 1

            if is_end:
                X = shuffle(X)
                bbatch_index = 0
            _, bloss, iloss, dloss = self.session.run([self.opt, self.loss_bou, self.loss_int, self.loss_div], \
                    feed_dict={self.sol_int_x: _fxy[:, 0].reshape((self.batch_size, 1)), \
                                self.sol_int_y: _fxy[:, 1].reshape((self.batch_size, 1)), \
                                self.sol_bou_x: _u1.reshape((self.batch_size, 1)), \
                                self.sol_bou_y: _u2.reshape((self.batch_size, 1)), \
                                self.X: _batch, \
                                self.X_boundary: _batch_bou, \
                                self.X_press: _batch, \
                                self.learning_rate: lr})
            
            ########## record loss ############
            ls_int.append(np.mean(np.squeeze(iloss)))
            ls_bou.append(np.mean(np.squeeze(bloss)))
            ls_div.append(np.mean(np.squeeze(dloss)))
            ls_total.append(ls_int[-1] + ls_bou[-1] + ls_div[-1])

            uh = self.session.run(self.model_velocity_int, feed_dict={self.X: vis_points})
            uhref = self.exact_solution(vis_points)
            l2 = 0
            for i in range(2):
                l2 += np.sqrt(np.mean((uh[:, i]-uhref[:, i])**2))
            ls_l2.append(l2)
            ########## record loss ############
            if it > 0 and it % vis_each_iters == 0:
                self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_{}.png'.format(it)))
            
        
            if it % 10 == 0:
                print("Iteration={}, Loss int: {}, Loss bou: {}, Loss div: {}, Loss total: {}, L2 error: {}".format(\
                        it, ls_int[-1], ls_bou[-1], ls_div[-1], ls_total[-1], ls_l2[-1]))
        # self.visualize_surface(t = 1, meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_final.png'))
        visualize_loss_error(ls_int, path = os.path.join(exp_folder, 'Loss_INT.png'), y_name = 'Loss int')
        visualize_loss_error(ls_bou, path = os.path.join(exp_folder, 'Loss_boundary.png'), y_name = 'Loss bou')
        visualize_loss_error(ls_div, path = os.path.join(exp_folder, 'Loss_div.png'), y_name = 'Loss div')
        visualize_loss_error(ls_total, path = os.path.join(exp_folder, 'Loss_total.png'), y_name = 'Loss total')
        visualize_loss_error(ls_l2, path = os.path.join(exp_folder, 'L2_Error.png'), y_name = 'L2 error')
        self.saver.save(self.session, os.path.join(exp_folder, 'model.ckpt'))
