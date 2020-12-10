import tensorflow as tf 
import numpy as np 

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

class NNPDE2:
    def __init__(self, batch_size, N, refn):
        self.rloss = []
        self.rbloss = []
        self.rl2 = []

        self.refn = refn  # reference points
        x = np.linspace(0, 1, refn)
        y = np.linspace(0, 1, refn)
        self.X, self.Y = np.meshgrid(x, y)
        self.refX = np.concatenate([self.X.reshape((-1, 1)), self.Y.reshape((-1, 1))], axis=1)

        self.batch_size = batch_size  # batchsize
        self.N = N # number of dense layers

        self.x = tf.placeholder(tf.float64, (None, 2)) # inner data
        self.x_b = tf.placeholder(tf.float64, (None, 2)) # boundary data

        self.u_b = self.bsubnetwork(self.x_b, False)
        self.u = self.bsubnetwork(self.x, True) + self.B(self.x) * self.subnetwork(self.x, False)

        self.bloss = tf.reduce_sum((self.tfexactsol(self.x_b)-self.u_b)**2)
        self.loss = self.loss_function()

        self.ploss = self.point_wise_loss()



        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "boundary")
        self.opt1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.bloss,var_list=var_list1)
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inner")
        self.opt2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss, var_list=var_list2)
        self.init = tf.global_variables_initializer()


    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def exactsol(self, x, y):
        raise NotImplementedError

    def tfexactsol(self, x):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    # end modification

    def subnetwork(self, x, reuse = False):
        with tf.variable_scope("inner"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="dense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="last", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def bsubnetwork(self, x, reuse = False):
        with tf.variable_scope("boundary"):
            for i in range(self.N):
                x = tf.layers.dense(x, 256, activation=tf.nn.tanh, name="bdense{}".format(i), reuse=reuse)
            x = tf.layers.dense(x, 1, activation=None, name="blast", reuse=reuse)
            x = tf.squeeze(x, axis=1)
            assert_shape(x, (None,))
        return x

    def point_wise_loss(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.abs(deltah - delta)
        assert_shape(res, (None,))
        return res

    def plot_exactsol(self):
        Z = self.exactsol(self.X, self.Y)
        ax = self.fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                        linewidth=0, antialiased=False, alpha=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        bX = np.zeros((4*self.batch_size, 2))
        bX[:self.batch_size,0] = np.random.rand(self.batch_size)
        bX[:self.batch_size,1] = 0.0

        bX[self.batch_size:2*self.batch_size, 0] = np.random.rand(self.batch_size)
        bX[self.batch_size:2*self.batch_size, 1] = 1.0

        bX[2*self.batch_size:3*self.batch_size, 0] = 0.0
        bX[2*self.batch_size:3*self.batch_size, 1] = np.random.rand(self.batch_size)

        bX[3*self.batch_size:4*self.batch_size, 0] = 1.0
        bX[3 * self.batch_size:4 * self.batch_size, 1] = np.random.rand(self.batch_size)

        bloss = sess.run([self.bloss], feed_dict={self.x_b: bX})[0]
        # if the loss is small enough, stop training on the boundary
        if bloss>1e-5:
            for _ in range(5):
                _, bloss = sess.run([self.opt1, self.bloss], feed_dict={self.x_b: bX})

        X = np.random.rand(self.batch_size, 2)
        _, loss = sess.run([self.opt2, self.loss], feed_dict={self.x: X})

        ########## record loss ############
        self.rbloss.append(bloss)
        self.rloss.append(loss)
        uh = sess.run(self.u, feed_dict={self.x: self.refX})
        Z = uh.reshape((self.refn, self.refn))
        uhref = self.exactsol(self.X, self.Y)
        self.rl2.append( np.sqrt(np.mean((Z-uhref)**2)) )
        ########## record loss ############

        if i % 10 == 0:
            print("Iteration={}, bloss = {}, loss= {}, L2={}".format(i, bloss, loss, self.rl2[-1]))

    def visualize(self, sess, showonlysol=False, i=None, savefig=None):

        x = np.linspace(0, 1, self.refn)
        y = np.linspace(0, 1, self.refn)
        [X, Y] = np.meshgrid(x, y)

        uh = sess.run(self.u, feed_dict={self.x: self.refX})
        Z = uh.reshape((self.refn, self.refn))

        uhref = self.exactsol(X, Y)

        def draw():
            self.fig = plt.figure()
            ax = self.fig.gca(projection='3d')

            if not showonlysol:
                ax.plot_surface(X, Y, uhref, rstride=1, cstride=1, cmap=cm.autumn,
                                linewidth=0, antialiased=False, alpha=0.3)

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                            linewidth=0, antialiased=False, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1.1)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            if i:
                plt.title("Iteration {}".format(i))
            if savefig:
                plt.savefig("{}/fig{}".format(savefig,0 if i is None else i))


class ProblemBLSingularity_BD(NNPDE2):
    def __init__(self, batch_size, N, refn):
        self.alpha = 0.6
        NNPDE2.__init__(self,batch_size, N, refn)

    def exactsol(self, x, y):
        return y**0.6

    def tfexactsol(self, x):
        return tf.pow(x[:,1],0.6)

    def f(self, x):
        return self.alpha*(self.alpha-1)*x[:,1]**(self.alpha-2)

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        delta = tf.clip_by_value(delta, -1e2, 1e2)
        deltah = tf.clip_by_value(deltah, -1e2, 1e2)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res



if __name__ == '__main__':
    dir = 'p3'
    npde = ProblemBLSingularity_BD(64, 3, 50) # works very well
    with tf.Session() as sess:
        sess.run(npde.init)
        for i in range(10000):
            if( i>1000 ):
                break
            npde.train(sess, i)
            if i%50==0:
                npde.visualize(sess, False, i=i, savefig=dir)