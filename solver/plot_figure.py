import matplotlib.pyplot as plt
import pylab

def plot_figure_1d(x, u, u_exact):

    n = u.shape[0]
    m = u.shape[1]
    x = x.reshape((m*n), order='F')
    u = u.reshape((m * n), order='F')
    u_exact = u_exact.reshape((m * n), order='F')

    plt.plot(x, u, 'ro')
    plt.plot(x, u_exact, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ion()
    plt.show()