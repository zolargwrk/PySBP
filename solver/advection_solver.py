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

# advection_solver_2d(p, h, t0, tf, cfl=1, flux_type='Central', boundary_type=None)
# u = advection_solver_2d(2, 0.5, 0, 1, cfl=1, flux_type='Upwind', boundary_type='nPeriodic')


def advection_solver_sbp_2d_steady(p, h, nrefine=1, sbp_family='diagE', p_map=1, curve_mesh=False):
    dim = 2
    nface = dim + 1
    nfp = p + 1
    ns = int((p + 1) * (p + 2) / 2)
    # the rectangular domain
    bL = 0
    bR = 1
    bB = 0
    bT = 1

    upwind = 0 # upwind = 1 gives upwind flux
    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
    domain_type = 'notperiodic'
    if domain_type == 'notperiodic':
        btype = ['d', 'd', 'd', 'd']
    elif domain_type == 'periodic':
        btype = ['-', '-', '-', '-']

    ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map=p_map, curve_mesh=curve_mesh,
                                          domain_type=domain_type)
    adata = SimpleNamespace(**ass_data)
    errs_soln = list()
    errs_adj = list()
    errs_func = list()
    hs = list()
    nelems = list()
    nnodes_list = list()
    cond_nums = list()
    nnz_elems = list()
    eig_vals = list()

    # refine mesh
    for refine in range(0, nrefine):
        if refine == 0:
            mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
        else:
            # mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT, True)
            mesh = MeshTools2D.hrefine_uniform_2d(ass_data, bL, bR, bB, bT)

        # update assembled data for 2D implementation
        ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map, curve_mesh=curve_mesh,
                                              domain_type=domain_type)
        adata = SimpleNamespace(**ass_data)

        # extract variables from adata
        x = adata.x
        y = adata.y
        r = adata.r
        s = adata.s
        nelem = adata.nelem
        nnodes = adata.nnodes
        vx = adata.vx
        vy = adata.vy
        xf = adata.xf
        yf = adata.yf
        Lx = adata.Lx  # length of domain in the x direction (not Lambda)
        Ly = adata.Ly  # length of domain in the y direction (not Lambda)
        etov = adata.etov
        if domain_type == 'notperiodic':
            etoe = adata.etoe
            etof = adata.etof
        elif domain_type == 'periodic':
            etoe = adata.etoe_periodic
            etof = adata.etof_periodic
        # initialize solution vectors
        u = 0 * x

        # get operators on physical elements
        phy_data = MeshTools2D.map_operators_to_phy_2d(p, nelem, adata.H, adata.Dr, adata.Ds, adata.Er, adata.Es,
                                                       adata.R1, adata.R2, adata.R3, adata.B1, adata.B2, adata.B3,
                                                       adata.rx, adata.ry, adata.sx, adata.sy, adata.jac,
                                                       adata.surf_jac, adata.nx, adata.ny)
        phy = SimpleNamespace(**phy_data)


        # advection coefficients
        IB = np.block([np.eye(nfp)] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        LrB = IB
        LsB = IB

        # get the advection coefficient
        Ln1B  = phy.nxB[0] * LrB + phy.nyB[0] * LsB
        Ln2B  = phy.nxB[1] * LrB + phy.nyB[1] * LsB
        Ln3B  = phy.nxB[2] * LrB + phy.nyB[2] * LsB
        Ln = [Ln1B, Ln2B, Ln3B]

        # upwind SATs
        T1kkB = 1/2*(phy.RB[0].transpose(0, 2, 1) @ phy.BB[0] @ (Ln[0] - upwind*np.abs(Ln[0])) @ phy.RB[0])
        T2kkB = 1/2*(phy.RB[1].transpose(0, 2, 1) @ phy.BB[1] @ (Ln[1] - upwind*np.abs(Ln[1])) @ phy.RB[1])
        T3kkB = 1/2*(phy.RB[2].transpose(0, 2, 1) @ phy.BB[2] @ (Ln[2] - upwind*np.abs(Ln[2])) @ phy.RB[2])
        Tkk = [T1kkB, T2kkB, T3kkB]

        HB_inv = np.linalg.inv(phy.HB)
        # construct the system matrix
        A = (np.block(np.zeros((nelem * nnodes, nelem * nnodes)))).reshape((nelem, nelem, nnodes, nnodes))

        A_diag = (phy.DxB + phy.DyB) - HB_inv @ (Tkk[0] + Tkk[1] + Tkk[2])

        # add the diagonals of the SAT matrix
        for i in range(0, nelem):
            A[i, i, :, :] += A_diag[i, :, :]

        # subtract interface SATs added at boundary facets
        for i in range(0, len(adata.bgrpD)):
            elem = adata.bgrpD[i, 0]
            face = adata.bgrpD[i, 1]
            A[elem, elem, :, :] += HB_inv[elem, :, :] @ Tkk[face][elem, :, :]

        # add the interface SATs at the neighboring elements
        for i in range(0, nelem):
            if i != etoe[i, 0]:
                # facet 1
                face = 0
                elem = i
                nbr_elem = etoe[i, 0]
                nbr_face = etof[i, 0]
                A[i, nbr_elem, :, :] += HB_inv[elem, :, :] @ (1/2 * phy.RB[face].transpose(0, 2, 1)[elem, :, :] @ phy.BB[face][elem, :, :]
                                                           @ np.flipud(np.fliplr(Ln[nbr_face][nbr_elem, :, :] - upwind*np.abs(Ln[face][elem, :, :])))
                                                           @ np.flipud(np.fliplr(phy.RB[nbr_face][nbr_elem, :, :])))
            if i != etoe[i, 1]:
                # facet 2
                face = 1
                elem = i
                nbr_elem = etoe[i, 1]
                nbr_face = etof[i, 1]
                A[i, nbr_elem, :, :] += HB_inv[elem, :, :] @ (1/2 * phy.RB[face].transpose(0, 2, 1)[elem, :, :] @ phy.BB[face][elem, :, :]
                                                           @ np.flipud(np.fliplr(Ln[nbr_face][nbr_elem, :, :] - upwind*np.abs(Ln[face][elem, :, :])))
                                                           @ np.flipud(np.fliplr(phy.RB[nbr_face][nbr_elem, :, :])))
            if i != etoe[i, 2]:
                # facet 3
                face = 2
                elem = i
                nbr_elem = etoe[i, 2]
                nbr_face = etof[i, 2]
                A[i, nbr_elem, :, :] += HB_inv[elem, :, :] @ (1/2 * phy.RB[face].transpose(0, 2, 1)[elem, :, :] @ phy.BB[face][elem, :, :]
                                                           @ np.flipud(np.fliplr(Ln[nbr_face][nbr_elem, :, :] - upwind*np.abs(Ln[face][elem, :, :])))
                                                           @ np.flipud(np.fliplr(phy.RB[nbr_face][nbr_elem, :, :])))

        # add boundary SATs
        for i in range(0, len(adata.bgrpD)):
            elem = adata.bgrpD[i, 0]
            face = adata.bgrpD[i, 1]
            A[elem, elem, :, :] -= HB_inv[elem, :, :] @ (phy.RB[face].transpose(0, 2, 1)[elem, :, :] @ phy.BB[face][elem, :, :]
                                                                @ (Ln[face][elem, :, :] - upwind*np.abs(Ln[face][elem, :, :]))
                                                                @ phy.RB[face][elem, :, :])

        sD = (np.block(np.zeros((nelem * nnodes, nelem * nnodes)))).reshape((nelem, nelem, nnodes, nnodes))
        for i in range(0, len(adata.bgrpD)):
            elem = adata.bgrpD[i, 0]
            face = adata.bgrpD[i, 1]
            sD[elem, elem, :, :] += HB_inv[elem, :, :] @ (phy.RB[face].transpose(0, 2, 1)[elem, :, :] @ phy.BB[face][elem, :, :]
                                                              @ (Ln[face][elem, :, :] - upwind*np.abs(Ln[face][elem, :, :]))
                                                             @ phy.RB[face][elem, :, :])

    u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
    A_mat = (A.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nnodes)
    A_mat = sparse.csr_matrix(A_mat)
    sD_mat = (sD.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nnodes)
    sD_mat = sparse.csr_matrix(sD_mat)
    sDf = sD_mat @ u_exact.reshape(-1, 1, order='F')
    f = (np.pi * np.cos(np.pi * x) * np.sin(np.pi*y) + np.pi * np.cos(np.pi * y) * np.sin(np.pi*x)).reshape(-1, 1, order='F') + sDf
    u = spsolve(A_mat, f)
    u = u.reshape(nelem, nnodes).T

    err = np.linalg.norm(u - u_exact)

    plot_figure_2d(x, y, u_exact)
    plot_figure_2d(x, y, u)

    return u


u = advection_solver_sbp_2d_steady(2, 5, 1, 'omega')