'''
    Define Multi Layers Perceptron model 
    Author: DamDev
    Date: 06/12/2020
    Reference: Deep Learning for Partial Differential Equations CS230, Kailai Xu, Bella Shi, Shuyi Yin
'''
import tensorflow as tf
import numpy as np
from compute_differential import assert_shape, compute_delta_nd
from sklearn.utils import shuffle

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

class DeepLearningPDE(object):
    '''
        Template of DeepLearning PDE models with specific ONLY inner model (u = 0 on boundary)
    '''
    def __init__(self, d = 3, inner_hidden_layers = [128]):
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

    def train(self, X, batch_size = 32, epochs = 10):
        '''
            Training
        '''
        # Handle data
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
        print('========================================')
        for epoch in range(epochs):
            X = shuffle(X)
            sum_losses = 0
            for it in range(iter_per_epoch):        
                # Training inner
                batch_inner, is_end = self.get_batch(X, it, batch_size = batch_size)
                _, loss_inner = self.session.run([self.opt_inner, self.loss_inner], feed_dict={self.X: batch_inner})
                sum_losses += np.sum(np.array(loss_inner))
            avg_loss = sum_losses/self.total_samples
            print('Epoch {}/{},  PDE loss: {}'.format(epoch+1, epochs, avg_loss))



class DeepLearningPDE_inner_boundary(object):
    '''
        Template of DeepLearning PDE models with specific inner & boundary model
    '''
    def __init__(self, d = 3, inner_hidden_layers = [128], boundary_hidden_layers = [128]):
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

        # Multi-layers perceptron model with tanh activation function
        self.model_Boundary = create_mlp_model(self.X_boundary, hidden_layers = boundary_hidden_layers, name = 'boundary')
        self.model_PDE      = create_mlp_model(self.X, hidden_layers = inner_hidden_layers, name = 'inner')
        
        # Reuse model_Boundary & model_PDE to compute u_predict
        self.u = create_mlp_model(self.X, hidden_layers = boundary_hidden_layers, name = 'boundary', reuse = True) + \
                        self.B(self.X)*self.model_PDE
        
        # Loss
        self.loss_boundary = self.compute_boundary_loss()
        self.loss_inner    = self.compute_inner_loss()
        
        # Optimizer
        var_list_boundary = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt_boundary = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_boundary, var_list=var_list_boundary)
        var_list_inner    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt_inner    = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_inner, var_list=var_list_inner)
        
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

    def compute_inner_loss(self):
        '''
            Loss function for inner points
        '''
        delta_predict = compute_delta_nd(self.u, self.X, self.dimension)
        delta_groundtruth = self.f(self.X)
        res = tf.reduce_sum((delta_predict - delta_groundtruth) ** 2)
        assert_shape(res, ())
        return res

    def compute_boundary_loss(self):
        '''
            Loss function for points in boundary
        '''
        loss = tf.reduce_sum((self.exact_solution(self.X_boundary) - self.model_Boundary) ** 2)
        return loss

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

    def train(self, X, X_boundary, batch_size = 32, epochs = 10, iters_training_boundary = 3):
        '''
            Training
        '''
        # Counter index of boundary batch
        bbatch_index = 0
        # Handle data
        assert len(X.shape) == 2, "Invalid input: X must be 2-dims numpy array, current shape: {}".format(X.shape)
        self.total_samples, dim = X.shape
        self.total_samples_boundary, dim_boundary = X_boundary.shape
        assert (dim == self.dimension), "Dimension of X and model must be equal"
        assert (dim_boundary == self.dimension), "Dimension of X_boundary and model must be equal"
        if self.total_samples % batch_size == 0:
            iter_per_epoch = self.total_samples // batch_size
        else:
            iter_per_epoch = self.total_samples // batch_size + 1
        if self.total_samples_boundary % batch_size == 0:
            iter_per_epoch_boundary = self.total_samples_boundary // batch_size
        else:
            iter_per_epoch_boundary = self.total_samples_boundary // batch_size + 1
        # Show information
        print('========================================')
        print('Number training inner: {}'.format(self.total_samples))
        print('Number training boundary: {}'.format(self.total_samples_boundary))
        print('Dimension: {}'.format(dim))
        print('Iterations per epoch: {}'.format(iter_per_epoch))
        print('========================================')
        for epoch in range(epochs):
            X = shuffle(X)
            ls_boundary = 0
            ct_boundary = 0
            ls_inner = 0
            ct_inner = 0
            for it in range(iter_per_epoch):
                # Training boundary
                for i in range(iters_training_boundary):
                    batch_boundary, is_end = self.get_batch(X_boundary, bbatch_index, batch_size = batch_size)
                    _, boundary_loss = self.session.run([self.opt_boundary, self.loss_boundary], feed_dict={self.X_boundary: batch_boundary})
                    bbatch_index += 1
                    if is_end:
                        X_boundary = shuffle(X_boundary)
                        bbatch_index = 0
                    ls_boundary += boundary_loss
                    ct_boundary += len(batch_boundary)      
                # Training inner
                batch_inner, is_end = self.get_batch(X, it, batch_size = batch_size)
                _, loss_inner = self.session.run([self.opt_inner, self.loss_inner], feed_dict={self.X: batch_inner})
                ls_inner += loss_inner
                ct_inner += len(batch_inner)
            avg_boundary = ls_boundary / ct_boundary
            avg_inner    = ls_inner / ct_inner
            print('Epoch {}/{}, Boundary loss: {},  PDE loss: {}'.format(epoch+1, epochs, avg_boundary, avg_inner))

