import numpy as np
from src.ref_elem import Ref2D


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


    @staticmethod
    def rhs_advection_2d(u, time, x, y, ax, ay, Dr, Ds, vmapM, vmapP, vmapD, mapD, nelem, nfp,
                         rx, lift, fscale, nx, ny, u_initial_function=1, flux_type='Upwind'):
        nface = 3

        # phase speed in the normal direction
        an = ax*nx + ay*ny

        # set central or upwind flux parameter
        if flux_type == 'Central':
            k = 0.5
        else:
            k = 0.5 * (1 + an/abs(an))

        # convert matrix form of arrays into array form
        u = u.flatten(order='F')
        vmapM = vmapM.flatten(order='F')
        vmapP = vmapP.flatten(order='F')
        mapD = mapD.flatten(order='F')

        # evaluate difference in solution along interface for flux calculation
        du = np.zeros((nfp * nface * nelem), 1)
        u_star = k*u(vmapM) + (1-k)*u(vmapP)
        du = u(vmapM) - u_star

        # boundary condition
        if u_initial_function != 1:
            u0 = u_initial_function(mapD)
        else:
            u0 = 0

        uM = u(vmapD)
        du[mapD] = uM - (k[mapD]*uM + (1-k[vmapD])) * u0

        # calculate flux
        df = an*du

        # evaluate RHS
        ux, uy = Ref2D.gradient_2d(x, y, Dr, Ds, u)
        rhs = -(an*ux + ay*uy) + lift*(fscale*df)

        return rhs