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
from compute_differential import assert_shape, compute_delta_nd
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from visualize import visualize_loss_error

def create_mlp_model(x, hidden_layers: list, name: str, reuse = False):
    '''
        Create MLP model from given hidden layers (number of layers & number of nodes each layer)
    '''
    with tf.variable_scope(name):
        for layer_id, layer_nodes in enumerate(hidden_layers):
            x = tf.layers.dense(x, layer_nodes, activation=tf.nn.tanh, name="dense{}".format(layer_id), reuse=reuse)
        x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)
        x = tf.squeeze(x, axis=1)
        assert_shape(x, (None,))
    return x

class LaplaceZeroBoundarySolver(object):
    '''
        Template of DeepLearning PDE models with specific ONLY inner model (u = 0 on boundary)
    '''
    def __init__(self, d = 3, inner_hidden_layers = [128], meshgrid = None):
        '''
            Init template with default config
                delta(u) = f
                u|dO = g_D
            -> solver u:= g_D + B(x, y).PDE(x, y, w2) 
                            g_D: Boundary g_D = dD
                            B: = 0 in dO
                            PDE: PDE deep learning model
        '''
        # Config
        self.dimension   = d
        self.X           = tf.placeholder(tf.float64, (None, self.dimension))
        self.meshgrid = meshgrid

        # Multi-layers perceptron model with tanh activation function
        self.model_PDE      = create_mlp_model(self.X, hidden_layers = inner_hidden_layers, name = 'inner')
        
        # Reuse model_Boundary & model_PDE to compute u_predict
        self.u = self.A(self.X) + self.B(self.X)*self.model_PDE
        
        # Loss
        self.loss_inner    = self.compute_inner_loss()
        
        # Optimizer
        var_list_inner    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt_inner    = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_inner, var_list=var_list_inner)
        # Session
        self.session = tf.Session()

        # Initializer all variables
        self.session.run([tf.global_variables_initializer()])

    def A(self, X): # 0, or tensor
        raise NotImplementedError

    def B(self, X): # Must be tensor, not numpy
        raise NotImplementedError

    def f(self, X): # Must be tensor, not numpy
        raise NotImplementedError

    def exact_solution(self, X): # Must be numpy, not tensor
        raise NotImplementedError

    def compute_inner_loss(self):
        '''
            Loss function for inner points
        '''
        delta_predict = compute_delta_nd(self.u, self.X, self.dimension)
        delta_groundtruth = self.f(self.X)
        res = tf.reduce_sum((delta_predict - delta_groundtruth) ** 2)
        assert_shape(res, ())
        return res

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

    def visualize_surface(self, meshgrid, save_path, show_only_sol = False):
        '''
            Visualize 3D surface
        '''
        print('<!> Visualize surface with meshgrid')
        X, Y = meshgrid

        vis_points = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
        u_predict = self.session.run(self.u, feed_dict={self.X: vis_points})
        Z = u_predict.reshape((len(Y), len(X)))

        u_true = self.exact_solution(vis_points)
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

    def train(self, X, \
            batch_size = 32, \
            epochs = 10, \
            exp_folder = 'exp', \
            draw_each_iter = False, \
            vis_each_epoches = 20, \
            meshgrid = None):
        '''
            Training
            input:
                - draw_each_iter: get & draw loss value each iter (default is each epoch)
        '''
        # Create project experiment
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
        # Handle data
        if meshgrid is not None:
            [meshX, meshY] = meshgrid
            vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
        elif self.meshgrid is not None:
            [meshX, meshY] = self.meshgrid
            vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
        else:
            vis_points = None
            print('<!> Warning: visualize points not found, using training points only!')
        assert len(X.shape) == 2, "Invalid input: X must be 2-dims numpy array, current shape: {}".format(X.shape)
        self.total_samples, dim = X.shape
        assert (dim == self.dimension), "Dimension of X and model must be equal"
        if self.total_samples % batch_size == 0:
            iter_per_epoch = self.total_samples // batch_size
        else:
            iter_per_epoch = self.total_samples // batch_size + 1
    
        # Show information
        print('========================================')
        print('Number of training examples: {}'.format(self.total_samples))
        print('Dimension: {}'.format(dim))
        print('Iterations per epoch: {}'.format(iter_per_epoch))
        print('Experimental: {}'.format(exp_folder))
        print('========================================')
        list_losses = []
        list_l2_vis = []
        list_l2_train = []
        for epoch in range(epochs):
            X = shuffle(X)
            sum_losses = 0
            for it in range(iter_per_epoch):        
                # Training inner
                batch_inner, is_end = self.get_batch(X, it, batch_size = batch_size)
                _, loss_inner = self.session.run([self.opt_inner, self.loss_inner], feed_dict={self.X: batch_inner})
                sum_losses += np.sum(np.array(loss_inner))
                if draw_each_iter:
                    list_losses.append(np.mean(np.array(loss_inner)))
            avg_loss = sum_losses/self.total_samples
            # Testing L2 Loss
            l2_err_train = self.l2_error(X)
            list_l2_train.append(l2_err_train)

            if vis_points is not None:
                l2_err_vis = self.l2_error(vis_points)
                list_l2_vis.append(l2_err_vis)
                print('Epoch {}/{},  PDE loss: {}, L2 error (train): {}, L2 error (test): {}'.format(epoch+1, epochs, avg_loss, l2_err_train, l2_err_vis))
            else:
                print('Epoch {}/{},  PDE loss: {}, L2 error (train): {}'.format(epoch+1, epochs, avg_loss, l2_err_train))

            if not draw_each_iter:
                list_losses.append(avg_loss)
            if epoch > 0 and epoch % vis_each_epoches == 0:
                self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_{}.png'.format(epoch)))
 
        # print(list_losses)
        if vis_points is not None:
            self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_final.png'))
            visualize_loss_error(list_l2_vis, path = os.path.join(exp_folder, 'L2_error_test.png'))
            print('Best L2 = {} on epoch {}'.format(np.amin(list_l2_vis), np.argmin(list_l2_vis)))
        visualize_loss_error(list_l2_train, path = os.path.join(exp_folder, 'L2_error_train.png'))
        visualize_loss_error(list_losses, path = os.path.join(exp_folder, 'PDE_loss.png'))


class LaplaceBoundarySolver(object):
    '''
        Template of DeepLearning PDE models with specific inner & boundary model
    '''
    def __init__(self, d = 3, inner_hidden_layers = [256], boundary_hidden_layers = [256]):
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
        self.X           = tf.placeholder(tf.float64, (None, self.dimension))
        self.X_boundary  = tf.placeholder(tf.float64, (None, self.dimension))
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
        self.opt_boundary = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss_boundary, var_list=var_list_boundary)
        var_list_inner    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt_inner    = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss_inner, var_list=var_list_inner)
        
        # Session
        self.session = tf.Session()

        # Initializer all variables
        self.session.run([tf.global_variables_initializer()])
        
    def B(self, X):
        raise NotImplementedError

    def f(self, X):
        raise NotImplementedError

    def exact_solution(self, X):
        raise NotImplementedError

    def tf_exact_solution(self, X):
        raise NotImplementedError

    def compute_inner_loss(self):
        '''
            Loss function for inner points
        '''
        delta_predict = compute_delta_nd(self.u, self.X, self.dimension)
        delta_groundtruth = self.f(self.X)
        delta_predict = tf.clip_by_value(delta_predict, -100, 100)
        delta_groundtruth = tf.clip_by_value(delta_groundtruth, -100, 100)
        res = tf.reduce_sum((delta_predict - delta_groundtruth) ** 2)
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
        u_predict = self.session.run(self.u, feed_dict={self.X: vis_points})
        Z = u_predict.reshape((len(Y), len(X)))

        u_true = self.exact_solution(vis_points)
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
            meshgrid = None):
        '''
            Training combine two loss functions
        '''
        # Define sum of two loss functions & optimizer
        ## Rebuild boundary loss using normal X placeholder (old: X_boundary placeholder)
        del self.model_Boundary
        del self.u
        del self.loss_boundary
        self.model_Boundary = create_mlp_model(self.X, hidden_layers = self.boundary_hidden_layers, name = 'boundary_new')
        # Reuse model_Boundary & model_PDE to compute u_predict
        self.u = create_mlp_model(self.X, hidden_layers = self.boundary_hidden_layers, name = 'boundary_new', reuse = True) + \
                        self.B(self.X)*create_mlp_model(self.X, hidden_layers = self.inner_hidden_layers, name = 'inner_new', reuse = False)
        self.loss_boundary = tf.reduce_sum((self.tf_exact_solution(self.X) - self.model_Boundary) ** 2)
        self.loss_sumary   = self.loss_boundary + self.loss_inner
        self.opt_sumary = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_sumary)
        self.session.run([tf.global_variables_initializer()])
        # Combine data
        training_samples = np.concatenate([X, X_boundary], axis = 0)
        training_samples = shuffle(training_samples)
        # Visualize
        meshX, meshY = meshgrid
        vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
        # Training
        ls_boundary  = []
        ls_inner     = []
        ls_l2        = []
        ls_total     = []
        for it in range(steps):
            batch, is_end = self.get_random_batch(training_samples, batch_size = batch_size)

            _, bloss, iloss = self.session.run([self.opt_sumary, self.loss_boundary, self.loss_inner], \
                    feed_dict={self.X: batch})
            
            ########## record loss ############
            ls_boundary.append(bloss)
            ls_inner.append(iloss)
            ls_total.append(bloss + iloss)
            uh = self.session.run(self.u, feed_dict={self.X: vis_points})
            Z = uh.reshape((len(meshY), len(meshX)))
            uhref = self.exact_solution(vis_points)
            uhref = uhref.reshape((len(meshY), len(meshX)))
            ls_l2.append(np.sqrt(np.mean((Z-uhref)**2)) )
            ########## record loss ############
            if it > 0 and it % vis_each_iters == 0:
                self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_{}.png'.format(it)))
            
        
            if it % 10 == 0:
                print("Iteration={}, Total Loss: {}, Bounding Loss: {}, PDE Loss: {}, L2 error: {}".format(\
                        it, bloss + iloss, bloss, iloss, ls_l2[-1]))
        self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_final.png'))
        visualize_loss_error(ls_l2, path = os.path.join(exp_folder, 'L2_error.png'))
        visualize_loss_error(ls_boundary, path = os.path.join(exp_folder, 'Boundary_loss.png'))
        visualize_loss_error(ls_inner, path = os.path.join(exp_folder, 'PDE_loss.png'))


    def train(self, X, X_boundary, \
            batch_size = 32, \
            steps = 1000, \
            iters_training_boundary = 3, \
            exp_folder = 'exp', \
            vis_each_iters = 100, \
            meshgrid = None, \
            train_method = None):
        '''
            Training
            Input:
                - iters_training_boundary: number iterations of training boundary before training inner
        '''
        # Create project experiment
        assert train_method in ['COMBINE', 'SEPARATE'], "Training method should be COMBINE or SEPARATE"
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)

        if train_method == 'SEPARATE':
            meshX, meshY = meshgrid
            vis_points = np.concatenate([meshX.reshape((-1, 1)), meshY.reshape((-1, 1))], axis=1)
            bbatch_index = 0
            ls_boundary = []
            ls_inner = []
            ls_l2 = []
            for it in range(steps):
                batch_boundary, is_end = self.get_batch(X_boundary, bbatch_index, batch_size = batch_size)
                bbatch_index += 1
                if is_end:
                    X_boundary = shuffle(X_boundary)
                    bbatch_index = 0

                bloss = self.session.run([self.loss_boundary], feed_dict={self.X_boundary: batch_boundary})[0]
                # if the loss is small enough, stop training on the boundary
                if bloss > 1e-5:
                    for _ in range(iters_training_boundary):
                        _, bloss = self.session.run([self.opt_boundary, self.loss_boundary], feed_dict={self.X_boundary: batch_boundary})

                batch_inner = self.get_random_batch(X, batch_size = batch_size)
                _, loss = self.session.run([self.opt_inner, self.loss_inner], feed_dict={self.X: batch_inner})

                ########## record loss ############
                ls_boundary.append(bloss)
                ls_inner.append(loss)
                uh = self.session.run(self.u, feed_dict={self.X: vis_points})
                Z = uh.reshape((len(meshY), len(meshX)))
                uhref = self.exact_solution(vis_points)
                uhref = uhref.reshape((len(meshY), len(meshX)))
                ls_l2.append(np.sqrt(np.mean((Z-uhref)**2)) )
                ########## record loss ############
                if it > 0 and it % vis_each_iters == 0:
                    self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_{}.png'.format(it)))
                
            
                if it % 10 == 0:
                    print("Iteration={}, Bounding Loss: {}, PDE Loss: {}, L2 error: {}".format(it, bloss, loss, ls_l2[-1]))
            self.visualize_surface(meshgrid = meshgrid, save_path = os.path.join(exp_folder, 'surface_final.png'))
            visualize_loss_error(ls_l2, path = os.path.join(exp_folder, 'L2_error.png'))
            visualize_loss_error(ls_boundary, path = os.path.join(exp_folder, 'Boundary_loss.png'))
            visualize_loss_error(ls_inner, path = os.path.join(exp_folder, 'PDE_loss.png'))
        else:
            self.train_combine(X, X_boundary, \
                batch_size = batch_size, \
                steps = steps, \
                exp_folder = exp_folder, \
                vis_each_iters = vis_each_iters, \
                meshgrid = meshgrid)