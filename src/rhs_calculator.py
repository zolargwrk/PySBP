import numpy as np
from src.ref_elem import Ref2D
from mesh.mesh_tools import MeshTools2D


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
    def rhs_advection_2d(u, time_loc, x, y, ax, ay, Dr, Ds, vmapM, vmapP, bnodes, bnodesB, nelem, nfp,
                         btype, lift, fscale, nx, ny, u_bndry_fun=None, flux_type='Upwind'):
        nface = 3
        n = u.shape[0]  # n = (p+1)*(p+2)/2
        # phase speed in the normal direction
        an = ax*nx + ay*ny

        # set central or upwind flux parameter
        if flux_type == 'Central':
            k = 0.5*np.ones(an.shape)
        else:
            k = 0.5 * (1 + an/abs(an))

        # convert matrix form of arrays into array form
        u = u.flatten(order='F')
        vmapM = vmapM.flatten(order='F')
        vmapP = vmapP.flatten(order='F')
        an = an.flatten(order='F')
        k = k.flatten(order='F')

        # evaluate difference in solution along interface for flux calculation
        u_star = k*u[vmapM] + (1-k)*u[vmapP]
        du = u[vmapM] - u_star

        # set boundary conditions
        u0 = MeshTools2D.set_bndry(u, x, y, ax, ay, time_loc, btype, bnodes, u_bndry_fun)

        # get boundary types into list
        mapB, vmapB = MeshTools2D.bndry_list(btype, bnodes, bnodesB)

        # get solution difference at the boundary and calculate flux
        du[mapB] = u[vmapB] - (k[mapB]*u[vmapB] + (1-k[mapB])*u0[vmapB])
        df = an*du

        # reshape vectors in to number of nodes per element by number of elements matrix form
        x = x.reshape((n, nelem), order='F')
        y = y.reshape((n, nelem), order='F')
        u = u.reshape((n, nelem), order='F')
        df = df.reshape((nfp*nface, nelem), order='F')

        # evaluate RHS
        ux, uy = Ref2D.gradient_2d(x, y, Dr, Ds, u, u)
        rhs = -(ax*ux + ay*uy) + lift@(fscale*df)

        return rhs
