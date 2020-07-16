import numpy as np
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator1D, MeshGenerator2D
from solver.plot_figure import plot_figure_1d, plot_figure_2d, plot_conv_fig
from src.error_conv import calc_err, calc_conv
from types import SimpleNamespace
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt


def advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type='Central', boundary_type=None, n=1):

    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a,  nelem, n)
    errs = list()
    dofs = list()
    nelems = list()

    # refine mesh uniformly
    nrefine = 5   # number of uniform refinements
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type)
        else:
            mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)

        nelem = mesh['nelem']       # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n)

        x = rhs_data['x']
        n = rhs_data['n']   # degrees of freedom per element
        dofs.append(n * nelem)
        nelems.append(nelem)

        # nx = MeshTools1D.normals_1d(nelem)

        x = x.reshape((n, nelem), order='F')
        u = -np.sin(10*np.pi*x)

        def u_bndry_fun(a, t, xl, xr):
            if a >= 0:
                u0 = np.sin(10*np.pi*t)
                # u0 = -np.sin(10*np.pi*(xl-a*t))
            else:
                u0 = -np.sin(10*np.pi*(xr-a*t))
            return u0

        rhs_calculator = RHSCalculator.rhs_advection_1d
        self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type, boundary_type)
        u = TimeMarcher.low_storage_rk4_1d(self_time_marcher, 0.1, x, a)

        u_exact = -np.sin(10*np.pi*(x - a * tf))

        # error calculation
        rx = rhs_data['rx']
        h_mat = rhs_data['h_mat']
        err = calc_err(u, u_exact, rx, h_mat)
        errs.append(err)

    hs = (xr-xl)/(np.asarray(nelems))
    conv = calc_conv(hs, errs, 1, 4)
    np.set_printoptions(precision=3, suppress=False)
    print(np.asarray(conv))
    print(np.asarray(errs))

    plot_conv_fig(hs, errs, 1, 4)
    plot_figure_1d(x, u, u_exact)

    return u


def advection_solver_2d(p, h, t0, tf, cfl=1, flux_type='Upwind', boundary_type=None):

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h, -1, 1, -1, 1)

    # obtain all data necessary for the residual (RHS) calculation
    self_assembler = Assembler(p)
    rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    # refine mesh
    nrefine = 0
    for i in range(0, nrefine):
        mesh = MeshTools2D.hrefine_uniform_2d(rhs_data)
        rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    x = rhs_data['x']
    y = rhs_data['y']

    # set initial condition and wave speed constants
    ax = 1
    ay = 1
    u = np.sin(np.pi * x) * np.sin(np.pi*y)

    def u_bndry_fun(x, y, ax, ay, t):
        ub = np.sin(np.pi*(x - ax*t)) * np.sin(np.pi*(y - ay*t))
        return ub

    # set type of boundary: [left, right, bottom, top]
    btype = ['d', '-', 'd', '-']

    rhs_calculator = RHSCalculator.rhs_advection_2d
    self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type, boundary_type)
    u = TimeMarcher.low_storage_rk4_2d(self_time_marcher, p, x, y, btype, ax, ay, cfl)

    u_exact = np.sin(np.pi * (x-ax*tf)) * np.sin(np.pi*(y-ay*tf))
    err = np.linalg.norm((u - u_exact), 2)
    print(err)
    plot_figure_2d(x, y, u)

    # #--------------------------
    # rhs_calculator = RHSCalculator.rhs_advection_steady_2d
    # self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type, boundary_type)
    # u = TimeMarcher.low_storage_rk4_2d(self_time_marcher, p, x, y, btype, ax, ay, cfl)

    return


# advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type = 'Central')
# u = advection_solver_1d(2, 0, 1, 2, 0, 2, 1, 'CSBP', 'Upwind', 'nPeriodic', n=17)

#advection_solver_2d(p, h, t0, tf, cfl=1, flux_type='Central', boundary_type=None)
# u = advection_solver_2d(2, 0.5, 0, 1, cfl=1, flux_type='Upwind', boundary_type='nPeriodic')


def advection_solver_sbp_2d_steady(p, h, sbp_family='diagE'):
    dim = 2
    nface = dim + 1
    nfp = p + 1
    ns = int((p + 1) * (p + 2) / 2)
    # the rectangular domain
    bL = 0
    bR = 1
    bB = 0
    bT = 1

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
    btype = ['d', 'd', 'd', 'd']

    ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family)
    adata = SimpleNamespace(**ass_data)
    x = adata.x
    y = adata.y
    nelem = adata.nelem
    nnodes = adata.nnodes
    bgrp = adata.bgrp
    bgrpD = adata.bgrpD
    rx = adata.rx
    ry = adata.ry
    sx = adata.sx
    sy = adata.sy
    Dr = adata.Dr
    Ds = adata.Ds
    surf_jac = adata.surf_jac
    nx = adata.nx
    ny = adata.ny
    jac = adata.jac
    R1 = adata.R1
    R2 = adata.R2
    R3 = adata.R3
    B1 = adata.B1
    B2 = adata.B2
    B3 = adata.B3
    etoe = adata.etoe
    etof = adata.etof

    H = adata.H

    u = 0 * x

    # face id
    fid1 = np.arange(0, nfp)
    fid2 = np.arange(nfp, 2 * nfp)
    fid3 = np.arange(2 * nfp, 3 * nfp)

    # boundary group (obtain element number and facet number only)
    bgrp = np.vstack(bgrp)[:, 2:4]
    # Dirichlet boundary groups by facet
    bgrpD1 = bgrpD2 = bgrpD3 = []
    if len(bgrpD) != 0:
        bgrpD1 = bgrpD[bgrpD[:, 1] == 0, :]
        bgrpD2 = bgrpD[bgrpD[:, 1] == 1, :]
        bgrpD3 = bgrpD[bgrpD[:, 1] == 2, :]

    # get the geometric factors for each element (in rxB, B stands for Block)
    rxB = rx.T.reshape(nelem, nnodes, 1)
    ryB = ry.T.reshape(nelem, nnodes, 1)
    sxB = sx.T.reshape(nelem, nnodes, 1)
    syB = sy.T.reshape(nelem, nnodes, 1)

    # get volume and surface Jacobians for each elements
    jacB = jac.T.reshape(nelem, nnodes, 1)
    surf_jac1B = surf_jac[fid1, :].flatten(order='F').reshape(nelem, nfp, 1)
    surf_jac2B = surf_jac[fid2, :].flatten(order='F').reshape(nelem, nfp, 1)
    surf_jac3B = surf_jac[fid3, :].flatten(order='F').reshape(nelem, nfp, 1)

    # get the normal vectors on each facet
    # nx1B = nx[fid1, :].flatten(order='F').reshape(nelem, nfp, 1)
    # ny1B = ny[fid1, :].flatten(order='F').reshape(nelem, nfp, 1)
    #
    # nx2B = nx[fid2, :].flatten(order='F').reshape(nelem, nfp, 1)
    # ny2B = ny[fid2, :].flatten(order='F').reshape(nelem, nfp, 1)
    #
    # nx3B = nx[fid3, :].flatten(order='F').reshape(nelem, nfp, 1)
    # ny3B = ny[fid3, :].flatten(order='F').reshape(nelem, nfp, 1)

    nx1B = np.repeat(nx[fid1, :].flatten(order='F').reshape(nelem, nfp, 1)[:,0,:], nelem, axis=1).reshape(nelem, nelem, 1)
    ny1B = np.repeat(ny[fid1, :].flatten(order='F').reshape(nelem, nfp, 1)[:,0,:], nelem, axis=1).reshape(nelem, nelem, 1)

    nx2B = np.repeat(nx[fid2, :].flatten(order='F').reshape(nelem, nfp, 1)[:,0,:], nelem, axis=1).reshape(nelem, nelem, 1)
    ny2B = np.repeat(ny[fid2, :].flatten(order='F').reshape(nelem, nfp, 1)[:,0,:], nelem, axis=1).reshape(nelem, nelem, 1)

    nx3B = np.repeat(nx[fid3, :].flatten(order='F').reshape(nelem, nfp, 1)[:,0,:], nelem, axis=1).reshape(nelem, nelem, 1)
    ny3B = np.repeat(ny[fid3, :].flatten(order='F').reshape(nelem, nfp, 1)[:,0,:], nelem, axis=1).reshape(nelem, nelem, 1)

    # get the derivative operator on the physical elements and store it for each element
    DrB = np.block([Dr] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
    DsB = np.block([Ds] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
    DxB = rxB * DrB + sxB * DsB
    DyB = ryB * DrB + syB * DsB

    R1B = np.block([R1] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
    R2B = np.block([R2] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
    R3B = np.block([R3] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)

    RB = [R1B, R2B, R3B]

    # get volume norm matrix and its inverse on physical elements
    HB = jacB * np.block([H] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
    HB_inv = np.linalg.inv(HB)

    IB = np.block([np.eye(H.shape[0])] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)

    # get surface norm matrix for each facet of each element
    BB1 = surf_jac1B * np.block([B1] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
    BB2 = surf_jac2B * np.block([B2] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
    BB3 = surf_jac3B * np.block([B3] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)

    BB = [BB1, BB2, BB3]

    # advection coefficients
    LxB = IB
    LyB = IB

    # get advection coefficient in the reference coordinate
    LrB = jacB*rxB * LxB + jacB*ryB * LyB
    LsB = jacB*sxB * LxB + jacB*syB * LyB

    # get the advection coefficient
    Lgk1B = nx1B * LrB + ny1B * LsB
    Lgk2B = nx2B * LrB + ny2B * LsB
    Lgk3B = nx3B * LrB + ny3B * LsB
    Lgk = [Lgk1B, Lgk2B, Lgk3B]

    # get advection coefficient normal to the shared face
    Ln1B = RB[0] @ LrB @ RB[0].transpose(0, 2, 1) + RB[0] @ LsB @ RB[0].transpose(0, 2, 1)
    Ln2B = -RB[1] @ LrB @ RB[1].transpose(0, 2, 1)
    Ln3B = -RB[2] @ LsB @ RB[2].transpose(0, 2, 1)
    Ln = [Ln1B, Ln2B, Ln3B]

    # coefficients for symmetric SATs
    # T1kkB = 1 / 2 *(R1B.transpose(0, 2, 1) @ BB1 @ R1B @ Lgk[0])
    # T2kkB = 1 / 2 *(R2B.transpose(0, 2, 1) @ BB2 @ R2B @ Lgk[1])
    # T3kkB = 1 / 2 *(R3B.transpose(0, 2, 1) @ BB3 @ R3B @ Lgk[2])

    # upwind SATs
    T1kkB = 1/2*(R1B.transpose(0, 2, 1) @ BB1 @ R1B @ Lgk1B - R1B.transpose(0, 2, 1) @ np.abs(BB1 @ Ln1B) @ R1B)
    T2kkB = 1/2*(R2B.transpose(0, 2, 1) @ BB2 @ R2B @ Lgk2B - R2B.transpose(0, 2, 1) @ np.abs(BB2 @ Ln2B) @ R2B)
    T3kkB = 1/2*(R3B.transpose(0, 2, 1) @ BB3 @ R3B @ Lgk3B - R3B.transpose(0, 2, 1) @ np.abs(BB3 @ Ln3B) @ R3B)
    Tkk = [T1kkB, T2kkB, T3kkB]

    # construct the system matrix
    A = (np.block(np.zeros((nelem * nnodes, nelem * nnodes)))).reshape((nelem, nelem, nnodes, nnodes))

    A_diag = 1/2*(DxB @ LxB + DyB @ LyB) + 1/2*(rxB * LxB @ DrB + sxB * LxB @ DsB)\
             + 1/2*(ryB * LyB @ DrB + syB * LyB @ DsB) - HB_inv @ (Tkk[0] + Tkk[1] + Tkk[2])

    # add the diagonals of the SAT matrix
    for i in range(0, nelem):
        A[i, i, :, :] += A_diag[i, :, :]

    # subtract interface SATs added at boundary facets
    for i in range(0, bgrp.shape[0]):
        elem = bgrp[i, 0]
        face = bgrp[i, 1]
        A[elem, elem, :, :] += 1 * HB_inv[elem, :, :] @ Tkk[face][elem, :, :]

    # add the interface SATs at the neighboring elements
    for i in range(0, nelem):
        if i != etoe[i, 0]:
            # facet 1
            nbr_elem = etoe[i, 0]
            nbr_face = etof[i, 0]
            # A[i, nbr_elem, :, :] -= HB_inv[i, :, :] @ (1/2 * R1B.transpose(0, 2, 1)[i, :, :] @ BB1[i, :, :]
            #                                            @Ln[nbr_face][nbr_elem, :, :] @ RB[nbr_face][nbr_elem, :, :])
            A[i, nbr_elem, :, :] -= HB_inv[i, :, :] @ (1/2 * R1B.transpose(0, 2, 1)[i, :, :]
                                                       @ (BB1[i, :, :] @ Ln[nbr_face][nbr_elem, :, :]
                                                          + np.abs(BB1[i, :, :] @ Ln[nbr_face][nbr_elem, :, :]))
                                                       @ RB[nbr_face][nbr_elem, :, :])
        if i != etoe[i, 1]:
            # facet 2
            nbr_elem = etoe[i, 1]
            nbr_face = etof[i, 1]
            # A[i, nbr_elem, :, :] -= HB_inv[i, :, :] @ (1/2 * R2B.transpose(0, 2, 1)[i, :, :] @ BB2[i, :, :]
            #                                            @Ln[nbr_face][nbr_elem, :, :] @ RB[nbr_face][nbr_elem, :, :])
            A[i, nbr_elem, :, :] -= HB_inv[i, :, :] @ (1/2 * R2B.transpose(0, 2, 1)[i, :, :]
                                                       @ (BB2[i, :, :] @ Ln[nbr_face][nbr_elem, :, :]
                                                          + np.abs(BB2[i, :, :] @ Ln[nbr_face][nbr_elem, :, :]))
                                                       @ RB[nbr_face][nbr_elem, :, :])
        if i != etoe[i, 2]:
            # facet 3
            nbr_elem = etoe[i, 2]
            nbr_face = etof[i, 2]
            # A[i, nbr_elem, :, :] -= HB_inv[i, :, :] @ (1/2 * R3B.transpose(0, 2, 1)[i, :, :] @ BB3[i, :, :]
            #                                            @Ln[nbr_face][nbr_elem, :, :] @ RB[nbr_face][nbr_elem, :, :])
            A[i, nbr_elem, :, :] -= HB_inv[i, :, :] @ (1/2 * R3B.transpose(0, 2, 1)[i, :, :]
                                                       @ (BB3[i, :, :] @ Ln[nbr_face][nbr_elem, :, :]
                                                          + np.abs(BB3[i, :, :] @ Ln[nbr_face][nbr_elem, :, :]))
                                                       @ RB[nbr_face][nbr_elem, :, :])

    # add boundary SATs
    for i in range(0, len(bgrpD)):
        elem = bgrpD[i, 0]
        face = bgrpD[i, 1]
        # A[elem, elem, :, :] -= HB_inv[elem, :, :] @ (1/2*RB[face].transpose(0, 2, 1)[elem, :, :] @ BB[face][elem, :, :]
        #                                              @ Ln[face][elem, :, :] @ RB[face][elem, :, :])
        A[elem, elem, :, :] -= HB_inv[elem, :, :] @ (RB[face].transpose(0, 2, 1)[elem, :, :] @ BB[face][elem, :, :]
                                                         @ RB[face][elem, :, :] @ Lgk[face][elem, :, :]
                                                         - RB[face].transpose(0, 2, 1)[elem, :, :]
                                                         @ np.abs(BB[face][elem, :, :] @ Ln[face][elem, :, :])
                                                         @ RB[face][elem, :, :])

    f = (np.pi * np.cos(np.pi * x) * np.sin(np.pi*y) + np.pi * np.cos(np.pi * y) * np.sin(np.pi*x))
    A_mat = (A.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nnodes)
    A_mat = sparse.csr_matrix(A_mat)
    u = spsolve(A_mat, f.reshape((nnodes * nelem, 1), order='F'))
    u = u.reshape(nelem, nnodes).T
    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    err = np.linalg.norm(u - u_exact)

    plot_figure_2d(x, y, u_exact)
    plot_figure_2d(x, y, u)

    return u


# u = advection_solver_sbp_2d_steady(2, 0.5, 'diagE')