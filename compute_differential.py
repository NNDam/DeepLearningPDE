'''
    Compute differential
    Author: DamDev
    Date: 06/12/2020
    Reference: Deep Learning for Partial Differential Equations CS230, Kailai Xu, Bella Shi, Shuyi Yin
'''
import tensorflow as tf

def assert_shape(x, shape):
    '''
        Assert shape of tensor (for debugging easier)
    '''
    S = x.get_shape().as_list()
    if len(S)!=len(shape):
        raise Exception("Shape mismatch: {} vs {}".format(S, shape))
    for i in range(len(S)):
        if S[i]!=shape[i]:
            raise Exception("Shape mismatch: {} vs {}".format(S, shape))
            
def compute_delta(u, xy):
    '''
        Compute dxx_u + dyy_u
    '''
    grad = tf.gradients(u, xy)[0]
    g1 = tf.gradients(grad[:,0], xy)[0]
    g2 = tf.gradients(grad[:,1], xy)[0]
    delta = g1[:,0] + g2[:,1]
    assert_shape(delta, (None,))
    return delta

def compute_delta_nd(u, X, n):
    '''
        Compute dx1x1_u + dx2x2_u + ... + dxnxn_u (n-dimension)
            X = [x1, x2, ... xn]
    '''
    grad = tf.gradients(u, X)[0]
    g1 = tf.gradients(grad[:, 0], X)[0]
    delta = g1[:,0]
    for i in range(1,n):
        g = tf.gradients(grad[:,i], X)[0]
        delta += g[:,i]
    assert_shape(delta, (None,))
    return delta

def compute_dx(u, xy):
    '''
        Compute dx_u
    '''
    grad = tf.gradients(u, xy)[0]
    dx_u = grad[:,0]
    assert_shape(dx_u, (None,))
    return dx_u

def compute_dy(u, xy):
    '''
        Compute dy_u
    '''
    grad = tf.gradients(u, xy)[0]
    dy_u = grad[:,1]
    assert_shape(dy_u, (None,))
    return dy_u

def compute_dt(u, t):
    grad = tf.gradients(u, t)[0]
    dt_u = grad[:,0]
    assert_shape(dt_u, (None,))
    return dt_u