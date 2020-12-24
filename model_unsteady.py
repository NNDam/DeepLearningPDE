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

class UnsteadyEquationSolver(object):
    '''
        Template of DeepLearning PDE models with specific inner & boundary model
    '''
    def __init__(self, d = 2, hidden_layers = [256]):
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
        visualize_loss_error(ls_loss)



if __name__ == '__main__':
    import scipy.io
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')
    exp_folder = 'problem_7_4x64'

    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1

    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    X = np.concatenate([t, x, y], axis = 1)
    U = np.concatenate([u, v], axis = 1)
    model = UnsteadyEquationSolver(hidden_layers = [128, 128, 128, 128])
    model.train_combine(X, U, batch_size = 512, steps = 100, exp_folder = exp_folder)
    model.restore(exp_folder = exp_folder)
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.session.run(model.lambda_1)
    lambda_2_value = model.session.run(model.lambda_2)
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    # Plot Results
    plot_solution(X_star, u_pred, 1, save_path = os.path.join(exp_folder, 'solution_xu.png'))
    plot_solution(X_star, v_pred, 2, save_path = os.path.join(exp_folder, 'solution_xv.png'))
    plot_solution(X_star, p_pred, 3, save_path = os.path.join(exp_folder, 'solution_xp.png'))    
    plot_solution(X_star, p_star, 4, save_path = os.path.join(exp_folder, 'solution_xp_true.png'))
    plot_solution(X_star, p_star - p_pred, 5, save_path = os.path.join(exp_folder, 'diff_p.png'))
    
    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
     # Load Data
    data_vort = scipy.io.loadmat('cylinder_nektar_t0_vorticity.mat')
           
    x_vort = data_vort['x'] 
    y_vort = data_vort['y'] 
    w_vort = data_vort['w'] 
    modes = np.asscalar(data_vort['modes'])
    nel = np.asscalar(data_vort['nel'])    
    
    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
    
    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    ####### Row 0: Vorticity ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    plt.savefig('Row0.png')