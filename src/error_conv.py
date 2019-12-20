import numpy as np
import scipy.sparse as sparse

def calc_err(u, u_exact, rx, h_mat):
    rx_global = np.diag(1/rx[0, :], 0)  # geometric factor rx = 1/jac
    h_mat_global = sparse.block_diag([h_mat])  # concatenate norm matrix to form global
    rh = sparse.kron(rx_global, h_mat_global)

    e = (u-u_exact).reshape((u.shape[0]*u.shape[1], 1), order = 'F')
    err = np.sqrt((e.T @ (rh @ e))[0, 0])
    # err= np.sqrt(np.sum(((u-u_exact)**2).flatten(order='F')))
    return err

def calc_conv(dofs, errs, n1, n2=None):
    dofs = np.asarray(dofs)
    errs = np.asarray(errs)
    dofs_conv = dofs[n1:n2]
    errs_conv = errs[n1:n2]
    conv = np.abs(np.polyfit(np.log10(dofs_conv), np.log10(errs_conv), 1)[0])

    return conv