import numpy as np
from scipy import sparse
from src.ref_elem import Ref2D
from mesh.mesh_tools import MeshTools2D
from src.sats import SATs
import matplotlib.pyplot as plt


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

        du[0:] = (u_projF[vmapM][:, 0] - u_projF[vmapP][:, 0]) * \
                 ((a * nx_tempC - (1 - alpha) * np.abs(a * nx_tempC)) / 2)

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
            bndry = MeshTools2D.bndry_list(btype, bnodes, bnodesB)
            mapD = bndry['mapD']
            vmapD = bndry['vmapD']

            # get solution difference at the boundary and calculate flux
            du[mapD] = u[vmapD] - (k[mapD]*u[vmapD] + (1-k[mapD])*u0[vmapD])

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
                         flux_type, sat_type='dg_sat', boundary_type='nPeriodic', db_mat=None, d2_mat=None, b=None, app=1,
                         uD_left=None, uD_right=None, uN_left=None, uN_right=None):

        n = u.shape[0]
        # set variable coefficient
        if type(b)==int or type(b)==float:
            b = b*np.ones(n)

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
            du = SATs.diffusion_dg_sat_1d(u, 'u', tl, tr, vmapM, vmapP, nx,  mapI, mapO, uin, uout, rx, flux_type)

            # calculate the derivative of the solution q
            q = rx*(d_mat @ u) - lift @ (fscale * (nx.T*du))

            ux = rx * (d_mat @ u)

            # boundary conditions on q (Neumann)
            qin = q.flatten(order='F')[vmapI]
            qout = q.flatten(order='F')[vmapO]
            uxin = ux.flatten(order='F')[vmapI]
            uxout = ux.flatten(order='F')[vmapO]
            dq = SATs.diffusion_dg_sat_1d(q, 'q', tl, tr, vmapM, vmapP, nx,  mapI, mapO, qin, qout, rx, flux_type, du,
                                          ux, uxin, uxout)

            # calculate rhs
            # rhs = 1/rx*(h_mat @(rx*(d_mat @ q) - lift @ (fscale * (nx.T*dq))))
            rhs = (rx * (d_mat @ q) - lift @ (fscale * (nx.T * dq)))
        elif sat_type=='sbp_sat':
            sI, fB= SATs.diffusion_sbp_sat_1d(u, d_mat, d2_mat, h_mat, tl, tr, vmapM, vmapP, nx, mapI, mapO, uin, uout,
                                           rx, flux_type, boundary_type, db_mat, b, app, uD_left, uD_right, uN_left,
                                           uN_right)
            if app==2:
                rhs = rx*rx*(d2_mat @ u) - sI
            else:
                rhs = rx*rx*(d_mat @ (np.diag(b) @ d_mat @ u)) - sI

        return rhs, fB

    @staticmethod
    def rhs_poisson_1d(n, nelem, d_mat, h_mat, lift, tl, tr, nx, rx, fscale, vmapM, vmapP, mapI, mapO, vmapI, vmapO,
                         flux_type, sat_type='dg_sat', boundary_type='Periodic', db_mat=None, d2_mat=None, b=1, app=1,
                         uD_left=None, uD_right=None, uN_left=None, uN_right=None):
        """Computes the system matrix for the Poisson equation
        flux_type: specify the SAT type of interest, e.g., BR1
        sat_type: specify whether to implement SAT as in 'dg_sat' or 'sbp_sat' ways"""
        g = np.zeros((nelem * n, 1))
        A = np.zeros((nelem * n, nelem * n))
        for i in range(0, nelem * n):
            g[i] = 1
            gmat = g.reshape((n, nelem), order='F')
            Avec, fB = RHSCalculator.rhs_diffusion_1d(gmat, d_mat, h_mat, lift, tl, tr, nx, rx, fscale, vmapM, vmapP, mapI,
                                                  mapO, vmapI, vmapO, flux_type, sat_type, boundary_type, db_mat, d2_mat,
                                                  b, app, uD_left, uD_right, uN_left, uN_right)
            # eliminate very small numbers from the A matrix
            sm = np.abs(Avec.reshape((n * nelem, 1), order='F')) >= 1e-12
            sm = (sm.reshape((n, nelem), order='F')) * 1
            Avec = (sm * Avec).reshape((nelem * n, 1), order='F').flatten()
            A[:, i] = Avec
            g[i] = 0

        return A, fB

    @staticmethod
    def rhs_poisson_1d_steady(n, nelem, d_mat, h_mat, lift, tl, tr, nx, rx, fscale, vmapM, vmapP, mapI, mapO, vmapI, vmapO,
                              flux_type, sat_type='dg_sat', boundary_type='Periodic', db_mat=None, d2_mat=None, b=1, app=1,
                              uD_left=None, uD_right=None, uN_left=None, uN_right=None):
        """Computes the system matrix for the Poisson equation
        flux_type: specify the SAT type of interest, e.g., BR1
        sat_type: specify whether to implement SAT as in 'dg_sat' or 'sbp_sat' ways"""

        # set variable coefficient
        if type(b) == int or type(b) == float:
            b = b * np.ones(n)

        # scale the matrices (the ones given are for the reference element)
        d2_mat = rx[0, 0] * rx[0, 0] * d2_mat
        Ablock = [d2_mat]*nelem

        # get SBP SAT terms
        sI, fB = SATs.diffusion_sbp_sat_1d_steady(n, nelem, d_mat, d2_mat, h_mat, tl, tr, nx, rx,
                                                  flux_type, boundary_type, db_mat, b, app,
                                                  uD_left, uD_right, uN_left, uN_right)

        A1 = sparse.block_diag(Ablock)
        A = A1 - sI

        return A, fB

    @ staticmethod
    def rhs_advection_1d_steady(n, nelem, d_mat, h_mat, tl, tr, rx, a=1, uD_left=None, uD_right=None, flux_type='upwind'):

        # set variable coefficient
        if type(a) == int or type(a) == float:
            a = a * np.ones(n)
        a_mat = np.diag(a.flatten())
        # scale the matrices (the ones given are for the reference element)
        d_mat = rx[0, 0] * d_mat @ a_mat
        Ablock = [d_mat]*nelem

        # get SBP SAT terms
        sI, fB = SATs.advection_sbp_sats_1d_steady(n, nelem, h_mat, tl, tr, rx, a, uD_left, uD_right, flux_type)

        A1 = sparse.block_diag(Ablock)
        A = A1 + sI

        return A, fB

    @staticmethod
    def rhs_diffusion_2d(p, u, x, y, r, s, Dr, Ds, lift, nx, ny, rx, fscale, vmapM, vmapP, mapM, mapP, mapD, vmapD, mapN,
                         vmapN, nelem, nfp, surf_jac, jac, flux_type='BR1'):

        n = u.shape[0]  # total number of nodes, n = (p+1)*(p+2)/2
        nface = 3

        # convert matrix form of arrays into array form
        u0 = u.copy()
        u = u.flatten(order='F')
        nx = nx.flatten(order='F')
        ny = ny.flatten(order='F')
        vmapM = vmapM.flatten(order='F')
        vmapP = vmapP.flatten(order='F')

        # compute difference in u at interfaces
        du = u[vmapM] - u[vmapP]
        du[mapD] = 2*u[vmapD]

        # compute qx and qy
        dudx, dudy = Ref2D.gradient_2d(x, y, Dr, Ds, u0)

        # compute flux in u
        fluxxu = (nx*du/2).reshape(nfp*nface, nelem,order='F')
        fluxyu = (ny*du/2).reshape(nfp*nface, nelem,order='F')
        qx = dudx - lift @ (fscale*fluxxu)
        qy = dudy - lift @ (fscale*fluxyu)

        qx = qx.flatten(order='F')
        qy = qy.flatten(order='F')

        # compute minimum height of abutting elements
        hmin = np.min([2*(jac.flatten(order='F'))[vmapP]/((surf_jac.flatten(order='F'))[mapP]).T,
                      2*(jac.flatten(order='F'))[vmapM]/((surf_jac.flatten(order='F'))[mapM]).T], axis=0)
        tau = (n/hmin).flatten()

        # compute difference in q at interfaces
        dqx = qx[vmapM] - qx[vmapP]
        dqx[mapN] = 2*qx[vmapN]
        dqy = qy[vmapM] - qy[vmapP]
        dqy[mapN] = 2*qy[vmapN]
        dqx.reshape(nfp*nface, nelem)
        dqy.reshape(nfp*nface, nelem)

        # compute flux in q
        fluxq = 0.5*(nx*dqx + ny*dqy) + (tau*du)/2
        fluxq = fluxq.reshape(nfp * nface, nelem,order='F')

        # compute divergence
        divq = Ref2D.divergence_2d(x, y, Dr, Ds, qx.reshape((Dr.shape[1], nelem), order='F'), qy.reshape((Ds.shape[1], nelem), order='F'))

        # compute rhs
        rhs = divq - lift @ (fscale*fluxq)

        # compute mass matrix times solution
        v = Ref2D.vandermonde_2d(p, r, s)
        Mu = jac*(((np.linalg.inv(v)).T @ np.linalg.inv(v)) @ u0)

        return rhs, Mu

    @staticmethod
    def rhs_poisson_2d(p, u, x, y, r, s, Dr, Ds, lift, nx, ny, rx, fscale, vmapM, vmapP, mapM, mapP, mapD, vmapD, mapN,
                       vmapN, nelem, nfp, surf_jac, jac, flux_type='BR1'):
        n = u.shape[0]
        g = np.zeros((nelem * n, 1))
        # A = np.zeros((nelem * n, nelem * n))
        A = sparse.lil_matrix((nelem * n, nelem * n))
        M = sparse.lil_matrix((nelem * n, nelem * n))
        for i in range(0, nelem * n):
            g[i] = 1
            gmat = g.reshape((n, nelem), order='F')
            Avec, Mvec = RHSCalculator.rhs_diffusion_2d(p, gmat, x, y, r, s, Dr, Ds, lift, nx, ny, rx, fscale, vmapM,
                                                        vmapP, mapM, mapP, mapD, vmapD, mapN, vmapN, nelem, nfp,
                                                        surf_jac, jac, flux_type)
            # eliminate very small numbers from the A and M matrices
            sm = np.abs(Avec.reshape((n * nelem, 1), order='F')) >= 1e-12
            sm = (sm.reshape((n, nelem), order='F')) * 1
            Avec = (sm * Avec).reshape((nelem * n, 1), order='F')

            sm = np.abs(Mvec.reshape((n * nelem, 1), order='F')) >= 1e-12
            sm = (sm.reshape((n, nelem), order='F')) * 1
            Mvec = (sm * Mvec).reshape((nelem * n, 1), order='F')

            A[:, i] = sparse.lil_matrix(Avec)
            M[:, i] = sparse.lil_matrix(Mvec)
            g[i] = 0

        A = sparse.spmatrix.tocsc(A)
        M = sparse.spmatrix.tocsc(M)
        return A, M

