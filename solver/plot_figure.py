import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pylab


def plot_figure_1d(x, u, u_exact=None):

    n = u.shape[0]
    m = u.shape[1]
    x = x.reshape((m*n), order='F')
    u = u.reshape((m * n), order='F')
    u_exact = u_exact.reshape((m * n), order='F')

    plt.plot(x, u, 'ro')
    plt.plot(x, u_exact, '-k*')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ion()
    plt.show()
    return

def plot_conv_fig(dofs, errs, n1, n2=None):

    # calculate convergence using line fitting
    dofs_conv = dofs[n1:n2]
    errs_conv = errs[n1:n2]
    conv = np.polyfit(np.log(dofs_conv), np.log(errs_conv), 1)
    convpoly = np.poly1d(conv)
    # define polynomial function
    yfit = lambda x: np.exp(convpoly(np.log(x)))

    # plt.loglog(dofs[-2:], yfit(dofs[-2:]), 'k')
    plt.loglog(dofs_conv, yfit(dofs_conv), 'k')
    plt.loglog(dofs, errs, 'ro')
    plt.xlabel('1/dof')
    plt.ylabel('error')
    # plt.ion()
    plt.show()
    return

def plot_figure_2d(x, y, u_exact):
    x = (x.flatten(order='F')).T
    y = (y.flatten(order='F')).T
    u_exact = u_exact.flatten(order='F').T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(x, y, u_exact, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()
    return
