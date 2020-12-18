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

def create_mlp_model(X, hidden_layers: list, name: str, reuse = False):
    '''
        Create MLP model from given hidden layers (number of layers & number of nodes each layer)
    '''
    with tf.variable_scope(name):
        for layer_id, layer_nodes in enumerate(hidden_layers):
            X = tf.layers.dense(X, layer_nodes, activation=tf.nn.tanh, name="dense{}".format(layer_id), reuse=reuse)
        X = tf.layers.dense(X, 1, activation=None, name="last", reuse=reuse)
        X = tf.squeeze(X, axis=1)
        assert_shape(X, (None,))
    return X

class HeatEquationSolver(object):
    '''
        Template of DeepLearning PDE models with specific inner & boundary model
    '''
    def __init__(self, d = 2, inner_hidden_layers = [256], boundary_hidden_layers = [256]):
        '''
            Init template with default config
                delta(u) = f
                u|dO = g_D
            -> solver u:= Boundary(x, y, w1) + B(x, y).PDE(x, y, w2) 
                            Boundary: boundary deep learning model (Boundary = g_D in dD)
                            B: = 0 in dO
                            PDE: PDE deep learning model
        '''
        # Config
        self.dimension   = d
        self.X           = tf.placeholder(tf.float64, (None, self.dimension + 1))
        self.X_boundary  = tf.placeholder(tf.float64, (None, self.dimension + 1))
        self.learning_rate  = tf.placeholder(tf.float64)
        self.boundary_hidden_layers = boundary_hidden_layers
        self.inner_hidden_layers = inner_hidden_layers

        # Multi-layers perceptron model with tanh activation function
        self.model_Boundary = create_mlp_model(self.X_boundary, hidden_layers = boundary_hidden_layers, name = 'boundary')
        # self.model_PDE      = create_mlp_model(self.X, hidden_layers = inner_hidden_layers, name = 'inner')
        
        # Reuse model_Boundary & model_PDE to compute u_predict
        self.u = create_mlp_model(self.X, hidden_layers = boundary_hidden_layers, name = 'boundary', reuse = True) + \
                        self.B(self.X)*create_mlp_model(self.X, hidden_layers = inner_hidden_layers, name = 'inner', reuse = False)
        
        # Loss
        self.loss_boundary = self.compute_boundary_loss()
        self.loss_inner    = self.compute_inner_loss()
        
        # Optimizer
        var_list_boundary = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt_boundary = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_boundary, var_list=var_list_boundary)
        var_list_inner    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt_inner    = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_inner, var_list=var_list_inner)
        
        # Session
        self.session = tf.Session()

        # Initializer all variables
        self.session.run([tf.global_variables_initializer()])
        
        # Saver
        self.saver = tf.train.Saver()

    def B(self, X, t):
        raise NotImplementedError

    def f(self, X, t):
        raise NotImplementedError

    def exact_solution(self, X, t):
        raise NotImplementedError

    def tf_exact_solution(self, X, t):
        raise NotImplementedError

    def compute_inner_loss(self):
        '''
            Loss function for inner points
        '''
        grad = tf.gradients(self.u, self.X)[0]
        g1 = tf.gradients(grad[:, 1], self.X)[0]
        g2 = tf.gradients(grad[:, 2], self.X)[0]
        _predict =  grad[:, 0] - g1[:, 1] - g2[:, 2]
        _groundtruth = self.f(self.X)
        # _predict = tf.clip_by_value(_predict, -100, 100)
        # _groundtruth = tf.clip_by_value(_groundtruth, -100, 100)
        res = tf.reduce_sum((_predict - _groundtruth) ** 2)
        assert_shape(res, ())
        return res

    def compute_boundary_loss(self):
        '''
            Loss function for points in boundary
        '''
        loss = tf.reduce_sum((self.tf_exact_solution(self.X_boundary) - self.model_Boundary) ** 2)
        return loss

    def l2_error(self, X):
        '''
            Compute L2 error
        '''
        u_predict = self.session.run([self.u], feed_dict = {self.X: X})
        u_predict = np.squeeze(u_predict)
        u_true    = self.exact_solution(X)
        m         = len(X)
        l2_err    = np.sqrt(np.sum((u_true - u_predict)**2)/m)
        return l2_err

    def get_batch(self, X, idx, batch_size = 32):
        '''
            Get index of sample for generating batch
        '''
        total_samples = len(X)
        is_end = False
        flag_start = idx*batch_size
        flag_end   = (idx+1)*batch_size
        if flag_end >= total_samples:
            flag_end = total_samples
            is_end = True
        return X[flag_start: flag_end], is_end

    def get_random_batch(self, X, batch_size = 32):
        '''
            Get index of sample for generating batch
        '''
        temp = shuffle(X)
        return temp[:batch_size]

    def visualize_surface(self, t, meshgrid, save_path, show_only_sol = False):
        '''
            Visualize 3D surface
        '''
        print('<!> Visualize surface with meshgrid')
        X, Y = meshgrid

        vis_points = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
        list_t     = np.ones((len(vis_points), 1))*t
        inpX          = np.concatenate([list_t, vis_points], axis = 1)

        u_predict = self.session.run(self.u, feed_dict={self.X: inpX})
        Z = u_predict.reshape((len(Y), len(X)))

        u_true = self.exact_solution(inpX)
        u_true = u_true.reshape((len(Y), len(X)))
        plt.clf()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if not show_only_sol:
            ax.plot_surface(X, Y, u_true, rstride=1, cstride=1, cmap=cm.autumn,
                            linewidth=0, antialiased=False, alpha=0.3)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                        linewidth=0, antialiased=False, alpha=0.8)
        ax.set_xlim(np.amin(X), np.amax(X))
        ax.set_ylim(np.amin(Y), np.amax(Y))
        ax.set_zlim(np.amin(Z) - 0.1, np.amax(Z + 0.1))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(save_path)

    def train_combine(self, X, X_boundary, \
            batch_size = 32, \
            steps = 1000, \
            exp_folder = 'exp', \
            vis_each_iters = 100, \
            meshgrid = None, \
            timespace = None, \
            lr_init = 0.01, \
            lr_scheduler = [4000, 6000, 8000]):
        '''
            Training combine two loss functions
        '''
        # Define sum of two loss functions & optimizer
        ## Rebuild boundary loss using normal X placeholder (old: X_boundary placeholder)

        self.model_Boundary = create_mlp_model(self.X, hidden_layers = self.boundary_hidden_layers, name = 'boundary_new')
        # Reuse model_Boundary & model_PDE to compute u_predict
        self.u = create_mlp_model(self.X, hidden_layers = self.boundary_hidden_layers, name = 'boundary_new', reuse = True) + \
                        self.B(self.X)*create_mlp_model(self.X, hidden_layers = self.inner_hidden_layers, name = 'inner_new', reuse = False)
        self.loss_boundary = tf.reduce_sum((self.tf_exact_solution(self.X) - self.model_Boundary) ** 2)
        self.loss_sumary   = self.loss_boundary + self.loss_inner

        self.opt_sumary = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_sumary)
        self.session.run([tf.global_variables_initializer()])
        # Combine data
        training_samples = np.concatenate([X, X_boundary], axis = 0)
        training_samples = shuffle(training_samples)
        # Visualize
        meshX, meshY = meshgrid
        vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
        all_points = np.tile(vis_points, (len(timespace), 1))
        all_t      = np.repeat(timespace, len(vis_points), 0).reshape((-1, 1))
        all_points = np.concatenate([all_t, all_points], axis = 1)
        # Training
        ls_boundary  = []
        ls_inner     = []
        ls_l2        = []
        ls_total     = []
        bbatch_index = 0
        lr = lr_init
        for it in range(steps):
            if it in lr_scheduler:
                lr = lr / 10
            batch, is_end = self.get_batch(training_samples, bbatch_index, batch_size = batch_size)
            bbatch_index += 1
            if is_end:
                training_samples = shuffle(training_samples)
                bbatch_index = 0
            _, bloss, iloss = self.session.run([self.opt_sumary, self.loss_boundary, self.loss_inner], \
                    feed_dict={self.X: batch, self.learning_rate: lr})
            
            ########## record loss ############
            ls_boundary.append(bloss)
            ls_inner.append(iloss)
            ls_total.append(bloss + iloss)

            uh = self.session.run(self.u, feed_dict={self.X: all_points, self.learning_rate: lr})
            uhref = self.exact_solution(all_points)
            ls_l2.append(np.sqrt(np.mean((uh-uhref)**2)))
            ########## record loss ############
            if it > 0 and it % vis_each_iters == 0:
                self.visualize_surface(t = 1, meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_{}.png'.format(it)))
            
        
            if it % 10 == 0:
                print("Iteration={}, Total Loss: {}, Bounding Loss: {}, PDE Loss: {}, L2 error: {}".format(\
                        it, bloss + iloss, bloss, iloss, ls_l2[-1]))
        self.visualize_surface(t = 1, meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_final.png'))
        visualize_loss_error(ls_l2, path = os.path.join(exp_folder, 'L2_error.png'), y_name = 'L2 error')
        visualize_loss_error(ls_boundary, path = os.path.join(exp_folder, 'Boundary_loss.png'), y_name = 'Loss boundary')
        visualize_loss_error(ls_inner, path = os.path.join(exp_folder, 'PDE_loss.png'), y_name = 'Loss PDE')
        self.saver.save(self.session, os.path.join(exp_folder, 'model.ckpt'))

    def train(self, X, X_boundary, \
            batch_size = 32, \
            steps = 1000, \
            iters_training_boundary = 3, \
            exp_folder = 'exp', \
            vis_each_iters = 100, \
            meshgrid = None, \
            train_method = None, \
            timespace = None, \
            lr_init = 0.01, \
            lr_scheduler = [4000, 6000, 8000]):
        '''
            Training
            Input:
                - iters_training_boundary: number iterations of training boundary before training inner
                - X = [t, x, y] and so on
        '''
        # Create project experiment
        assert train_method in ['COMBINE', 'SEPARATE'], "Training method should be COMBINE or SEPARATE"
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
        print(X.shape, X_boundary.shape)
        if train_method == 'SEPARATE':
            # Create list of point for calculate L2
            lr = lr_init
            meshX, meshY = meshgrid
            vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
            all_points = np.tile(vis_points, (len(timespace), 1))
            all_t      = np.repeat(timespace, len(vis_points), 0).reshape((-1, 1))
            all_points = np.concatenate([all_t, all_points], axis = 1)
            bbatch_index = 0
            ls_boundary = []
            ls_inner = []
            ls_l2 = []
            for it in range(steps):
                if it in lr_scheduler:
                    lr = lr / 10
                batch_boundary, is_end = self.get_batch(X_boundary, bbatch_index, batch_size = batch_size)
                bbatch_index += 1
                if is_end:
                    X_boundary = shuffle(X_boundary)
                    bbatch_index = 0
                bloss = self.session.run([self.loss_boundary], \
                    feed_dict={self.X_boundary: batch_boundary, self.learning_rate: lr})[0]
                # if the loss is small enough, stop training on the boundary
                if bloss > 1e-3:
                    for _ in range(iters_training_boundary):
                        _, bloss = self.session.run([self.opt_boundary, self.loss_boundary], \
                            feed_dict={self.X_boundary: batch_boundary, self.learning_rate: lr})

                batch_inner = self.get_random_batch(X, batch_size = batch_size)
                _, loss = self.session.run([self.opt_inner, self.loss_inner], feed_dict={self.X: batch_inner, self.learning_rate: lr})

                ########## record loss ############
                ls_boundary.append(bloss)
                ls_inner.append(loss)
                uh = self.session.run(self.u, feed_dict={self.X: all_points})
                uhref = self.exact_solution(all_points)
                ls_l2.append(np.sqrt(np.mean((uh-uhref)**2)))
                # ls_l2.append(0)
                ########## record loss ############
                if it > 0 and it % vis_each_iters == 0:
                    self.visualize_surface(t = 1, meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_{}.png'.format(it)))
                
            
                if it % 10 == 0:
                    print("Iteration={}, Bounding Loss: {}, PDE Loss: {}, L2 error: {}".format(it, bloss, loss, ls_l2[-1]))
            # self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_final.png'))
            visualize_loss_error(ls_l2, path = os.path.join(exp_folder, 'L2_error.png'), y_name = 'L2 error')
            visualize_loss_error(ls_boundary, path = os.path.join(exp_folder, 'Boundary_loss.png'), y_name = 'Loss boundary')
            visualize_loss_error(ls_inner, path = os.path.join(exp_folder, 'PDE_loss.png'), y_name = 'Loss inner')
            self.saver.save(self.session, os.path.join(exp_folder, 'model.ckpt'))
        else:
            self.train_combine(X, X_boundary, \
                batch_size = batch_size, \
                steps = steps, \
                exp_folder = exp_folder, \
                vis_each_iters = vis_each_iters, \
                meshgrid = meshgrid, \
                timespace = timespace, \
                lr_init = lr_init, \
                lr_scheduler = lr_scheduler)