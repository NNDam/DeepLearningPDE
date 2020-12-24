import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualize_loss_error(list_losses, path = 'loss.png', x_name = 'iter', y_name = 'loss'):
    '''
        Visualize loss & error
    '''
    plt.clf()
    list_losses = np.array(list_losses)
    Xbar = np.arange(len(list_losses))
    fig, ax = plt.subplots() 
    ax.plot(Xbar, list_losses)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(path)

    plt.clf()
    list_losses = np.array(list_losses)
    Xbar = np.arange(len(list_losses))
    fig, ax = plt.subplots() 
    ax.plot(Xbar, list_losses)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    ax.set_yscale('log')
    plt.savefig(path.split('.')[0] + '_log.png')


def visualize_f(meshgrid, func, save_path):
    '''
        Visualize 3D surface
    '''
    print('<!> Visualize surface with meshgrid')
    X, Y = meshgrid

    vis_points = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)

    u_true = func(vis_points)
    u_true = u_true.reshape((len(Y), len(X)))
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, u_true, rstride=1, cstride=1, cmap=cm.autumn,
                    linewidth=0, antialiased=False, alpha=0.3)

    ax.set_xlim(np.amin(X), np.amax(X))
    ax.set_ylim(np.amin(Y), np.amax(Y))
    ax.set_zlim(np.amin(u_true) - 0.1, np.amax(u_true + 0.1))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.savefig(save_path)