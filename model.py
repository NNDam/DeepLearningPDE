'''
    Define Multi Layers Perceptron model 
    Author: DamDev
    Date: 06/12/2020
    Reference: Deep Learning for Partial Differential Equations CS230, Kailai Xu, Bella Shi, Shuyi Yin
'''
import tensorflow as tf
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

# class DeepLearningPDE_basic(object):
#     '''
#         Template of DeepLearning PDE models for solving:
#         delta(u) = f
#         u| 
#     '''
#     def __init__(self, d = 3, hidden_layers = [128]):
#         '''
#             Init template with default config
#         '''
#         self.dimension = d
#         self.X = tf.placeholder(tf.float64, (None, self.dimension))
#         self.u = self.calculate_u(self.X)
#         self.

#     def compute_loss(self):
#         '''
#             L2-Loss function
#         '''
#         delta_predict = compute_delta_nd(self.u, self.X, self.dimension)
#         delta_groundtruth = self.f(self.X)
#         res = tf.reduce_sum((delta_predict - delta_groundtruth) ** 2)
#         assert_shape(res, ())
#         return res

#     def calculate_u(self):
#         raise NotImplementedError



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
        
        # Initializer all variables
        self.init = tf.global_variables_initializer()
        
        # Session
        self.session = tf.Session()

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

    def train(self, X, batch_size = 32, epochs = 10, ):
        '''
            Training
        '''
        assert len(X.shape) == 2, "Invalid input: X must be 2-dims numpy array, current shape: {}".format(X.shape)
        total_samples, dim = X.shape
        assert (dim == self.dimension), "Dimension of X and model must be equal"
        if total_samples % batch_size == 0:
            iter_per_epoch = total_samples // batch_size
        else:
            iter_per_epoch = total_samples // batch_size + 1
        print('========================================')
        print('Number training: {}'.format(total_samples))
        print('Dimension: {}'.format(dim))
        print('Iterations per epoch: {}'.format(iter_per_epoch))
        print('========================================')
        for epoch in range(epochs):
            X = shuffle(X)
            for it in range(iter_per_epoch):
                batch, is_end = self.get_batch(total_samples, it, batch_size = batch_size)
                batc
                print('Epoch {}/{}, Iter: {}, Boundary loss: {},  PDE loss: {}'.format(epoch, epochs, it, ))
