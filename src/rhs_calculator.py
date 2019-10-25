import numpy as np
from src.ref_elem import Ref2D
from mesh.mesh_tools import MeshTools2D
from src.sat_types import SATs


class RHSCalculator:

    @staticmethod
    def rhs_advection_1d(u, x, time, a, xl, xr, d_mat, vmapM, vmapP, mapI, mapO, vmapI, tl, tr,
                         rx, lift, fscale, nx, u_bndry_fun=None, flux_type='Upwind', boundary_type=None):

        if flux_type == 'Central':
            alpha = 1
        else:
            alpha = 0

        nface = 2
        n = u.shape[0]
        nelem = u.shape[1]

        # project u to the interfaces to get the u-u* at the faces
        u_proj = u.copy()
        u_projM = (u.T @ tl)[:, 0]
        u_projP = (u.T @ tr)[:, 0]
        u_proj[0, :] = u_projM
        u_proj[n - 1, :] = u_projP

        du = np.zeros((nface, nelem))
        du = du.reshape((nface * nelem, 1), order='F')

        # reshape some matrices and calculate du
        u_projF = u_proj.reshape((n * nelem, 1), order='F')
        nx_tempC = nx.reshape((nface * nelem, 1), order='C')

        du[0:] = (u_projF[vmapM][:, 0] - u_projF[vmapP][:, 0]) * (
                    (a * nx_tempC - (1 - alpha) * np.abs(a * nx_tempC)) / 2)

        if boundary_type != 'Periodic':
            # set dirichlet boundary condition
            u0 = u_bndry_fun(a, time, xl, xr)
            # set periodic boundary condition
            du[mapI] = (u_projF[vmapI] - u0) * (a * nx_tempC[mapI] - (1 - alpha) * np.abs(a * nx_tempC[mapI])) / 2
            du[mapO] = 0

        du = du.reshape((nface, nelem), order='F')
        rhs = (-a*rx)*(d_mat @ u) + lift @ (fscale * du)

        return rhs


    @staticmethod
    def rhs_advection_2d(u, time_loc, x, y, fx, fy, ax, ay, Dr, Ds, vmapM, vmapP, bnodes, bnodesB, nelem, nfp,
                         btype, lift, fscale, nx, ny, u_bndry_fun=None, flux_type='Upwind', boundary_type=None):
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

        if boundary_type != 'Periodic':
            # set boundary conditions
            u0 = MeshTools2D.set_bndry(u, x, y, ax, ay, time_loc, btype, bnodes, u_bndry_fun)

            # get boundary types into list
            mapB, vmapB = MeshTools2D.bndry_list(btype, bnodes, bnodesB)

            # get solution difference at the boundary and calculate flux
            du[mapB] = u[vmapB] - (k[mapB]*u[vmapB] + (1-k[mapB])*u0[vmapB])

        # calculate flux
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


    @staticmethod
    def rhs_diffusion_1d(u, d_mat, h_mat, lift, tl, tr, nx, rx, fscale, vmapM, vmapP, mapI, mapO, vmapI, vmapO,
                         flux_type, sat_type='dg_sat', boundary_type='Periodic', db_mat=None, d2_mat=None, b=1, app=1,
                         uD_left=None, uD_right=None, uN_left=None, uN_right=None):

        n = u.shape[0]
        # set variable coefficient
        if b==1:
            b = np.ones(n)

        # project u to the interfaces to get the u-u* at the faces
        u_proj = u.copy()
        u_projM = (u.T @ tl)[:, 0]
        u_projP = (u.T @ tr)[:, 0]
        u_proj[0, :] = u_projM
        u_proj[n - 1, :] = u_projP

        # boundary condition on solution u (Dirichlet)
        u_proj = np.reshape(np.array([u_projM, u_projP]), (2*len(u_projM), 1), order='F')
        uin = -u_proj[mapI]
        uout = -u_proj[mapO]

        if sat_type == 'dg_sat':
            # get solution differences at the interfaces and boundaries
            du = SATs.diffusion_dg_sat_1d(u, 'u', tl, tr, vmapM, vmapP, nx,  mapI, mapO, uin, uout, flux_type)

            # calculate the derivative of the solution q
            q = rx*(d_mat @ u) - lift @ (fscale * (nx.T*du))

            # boundary conditions on q (Neumann)
            qin = q.flatten(order='F')[vmapI]
            qout = q.flatten(order='F')[vmapO]
            dq = SATs.diffusion_dg_sat_1d(q, 'q', tl, tr, vmapM, vmapP, nx,  mapI, mapO, qin, qout, flux_type, du)

            # calculate rhs
            rhs = 1/rx*(h_mat @(rx*(d_mat @ q) - lift @ (fscale * (nx.T*dq))))
            # rhs = (rx * (d_mat @ q) - lift @ (fscale * (nx.T * dq)))
        elif sat_type=='sbp_sat':
            [sI, sB] = SATs.diffusion_sbp_sat_1d(u, d_mat, h_mat, tl, tr, vmapM, vmapP, nx, mapI, mapO, uin, uout,
                                           rx, flux_type, boundary_type, db_mat, b, app, uD_left, uD_right, uN_left,
                                           uN_right)
            if app==2:
                rhs = rx*rx*(d2_mat @ u) - sI - sB
            else:
                rhs = rx*rx*(d_mat @ (np.diag(b) @ d_mat @ u)) - sI - sB

        return rhs