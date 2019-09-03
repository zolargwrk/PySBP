import numpy as np


class RHSCalculator:

    @staticmethod
    def rhs_advection_1d(u, time, a, d_mat, vmapM, vmapP, mapI, mapO, vmapI, tl, tr,
                         rx, lift, fscale, nx, u_initial_function=1, flux_type='Upwind'):

        if flux_type == 'Central':
            alpha = 1
        else:
            alpha = 0

        if u_initial_function == 1:
            def u_init(arg1, arg2):
                return 0
        else:
            u_init = u_initial_function


        nface = 2
        n = u.shape[0]
        nelem = u.shape[1]

        # project u to the boundaries to get the u-u* at the faces
        u_proj = u.copy()
        u_projM = (u.T @ tl)[:, 0]
        u_projP = (u.T @ tr)[:, 0]
        u_proj[0, :] = u_projM
        u_proj[n-1, :] = u_projP

        du = np.zeros((nface, nelem))
        du = du.reshape((nface * nelem, 1), order='F')
        du[0:] = (u_proj.reshape((n*nelem, 1), order='F')[vmapM][:, 0] - u_proj.reshape((n*nelem, 1), order='F')[vmapP][:, 0])\
                 *((a*nx.reshape(nface*nelem, 1) - (1-alpha)*np.abs(a*nx.reshape(nface*nelem, 1)))/2)

        u0 = u_init(a, time)
        du[mapI] = (u_proj.reshape((n * nelem, 1), order='F')[vmapI] - u0)*(a*nx.reshape((nface * nelem, 1), order='F')[mapI] - (1 - alpha) * np.abs(a * nx.reshape((nface * nelem, 1), order='F')[mapI])) / 2
        du[mapO] = 0
        du = du.reshape((nface, nelem), order='F')

        rhs = (-a*rx)*(d_mat @ u) + lift @ (fscale * du)

        return rhs
