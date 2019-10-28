import numpy as np
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator2D, MeshGenerator1D
from solver.plot_figure import plot_figure_1d, plot_figure_2d, plot_conv_fig
from src.error_conv import calc_err, calc_conv
from types import SimpleNamespace
import matplotlib.pyplot as plt


def heat_1d(p, xl, xr, nelem, t0, tf, quad_type, flux_type='BR1', nrefine=1, boundary_type=None,
                        sat_type='dg_sat', a=1, b=1, n=1, app=1):

    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)
    errs = list()
    dofs = list()
    nelems = list()

    # boundary conditions
    uD_left = 0  # Dirichlet boundary at the left boundary
    uD_right = 0  # Dirichlet boundary at the right boundary
    uN_left = None  # Neumann boundary at the left boundary
    uN_right = None  # Neumann boundary at the right boundary

    # refine mesh uniformly
    # nrefine = 3  # number of uniform refinements
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        else:
            mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)

        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)

        x = rhs_data['x']
        n = rhs_data['n']  # degrees of freedom
        dofs.append(n*nelem)
        nelems.append(nelem)

        # nx = MeshTools1D.normals_1d(nelem)

        x = x.reshape((n, nelem), order='F')
        u = np.sin(x)

        rhs_calculator = RHSCalculator.rhs_diffusion_1d
        self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, None, flux_type, boundary_type,
                                        sat_type, app)
        u = TimeMarcher.low_storage_rk4_1d(self_time_marcher, 0.25, x, xl, a, b, uD_left, uD_right, uN_left, uN_right)

        u_exact = np.exp(-tf)*np.sin(x)

        # error calculation
        rx = rhs_data['rx']
        h_mat = rhs_data['h_mat']
        err = calc_err(u, u_exact, rx, h_mat)
        errs.append(err)

    plot_err = 1
    if plot_err == 1:
        conv_start = 1
        conv_end = nrefine
        hs = (xr - xl) / (np.asarray(nelems))
        conv = calc_conv(hs, errs, conv_start, conv_end)
        np.set_printoptions(precision=3, suppress=False)
        print(np.asarray(conv))
        print(np.asarray(errs))

        plot_conv_fig(hs, errs, conv_start, conv_end)

    plot_figure_1d(x, u, u_exact)

    return u


# heat_1d(p, xl, xr, nelem, t0, tf, quad_type, flux_type='BR1', nrefine, boundary_type=None, b=1, n=1):
# u = heat_1d(2, 0, 2*np.pi, 5, 0, 0.1, 'HGTL', 'BR2', 3, 'nPeriodic', 'sbp_sat', a=1, b=1, n=13, app=2)

# For Order-Matched and compatible operators (app=2) the HGT works with BR2 scheme. HGTL and CSBP do not work, Next time
#   -- check why for order-matched operator the HGTL and CSBP operators do not work
#   -- check why HGT does not work for BR1
# **--** implement the methods for the Poisson problem and see if the issues still exist


def poisson_1d(p, xl, xr, nelem, quad_type, flux_type='BR1', nrefine=1, boundary_type=None, sat_type='dg_sat',
               a=1, b=1, n=1, app=1):

    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)
    errs = list()
    dofs = list()
    nelems = list()

    # refine mesh uniformly
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        else:
            mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)

        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)

        rdata = SimpleNamespace(**rhs_data)
        dofs.append(n*nelem)
        nelems.append(nelem)

        # extract some information from rdata
        n = rdata.n
        x = (rdata.x).reshape((n, nelem), order='F')

        # boundary conditions
        uD_left = 0     # Dirichlet boundary at the left boundary
        uD_right = 0    # Dirichlet boundary at the right boundary
        uN_left = None     # Neumann boundary at the left boundary
        uN_right = None    # Neumann boundary at the right boundary

        g = np.zeros((nelem*n, 1))
        A = np.zeros((nelem*n, nelem*n))
        for i in range (0, nelem*n):
            g[i] = 1
            gmat = g.reshape((n, nelem), order='F')
            Avec = RHSCalculator.rhs_diffusion_1d(gmat, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx,
                                                rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                                rdata.vmapI, rdata.vmapO, flux_type, sat_type, boundary_type,
                                                rdata.db_mat, rdata.d2_mat, b, app, uD_left, uD_right, uN_left, uN_right)
            # eliminate very small numbers from the A matrix
            sm = np.abs(Avec.reshape((n * nelem, 1), order='F')) >= 1e-12
            sm = (sm.reshape((n, nelem), order='F'))*1
            Avec = (sm*Avec).reshape((nelem*n, 1), order='F').flatten()
            A[:, i] = Avec
            g[i] = 0

        # f = -1/rdata.rx * (rdata.h_mat @ np.sin(x))
        f = -np.sin(x)
        u = np.linalg.inv(A) @ f.reshape((n*nelem, 1), order='F')
        u_exact = (np.sin(x)).reshape((n*nelem, 1), order='F')

        # error calculation
        rx = rhs_data['rx']
        h_mat = rhs_data['h_mat']
        err = calc_err(u, u_exact, rx, h_mat)
        errs.append(err)

    plot_err = 1
    if plot_err == 1:
        conv_start = 1
        conv_end = nrefine
        hs = (xr - xl) / (np.asarray(nelems))
        conv = calc_conv(hs, errs, conv_start, conv_end)
        np.set_printoptions(precision=3, suppress=False)
        print(np.asarray(conv))
        print(np.asarray(errs))

        plot_conv_fig(hs, errs, conv_start, conv_end)

    # print(np.count_nonzero(A))
    # print(np.linalg.cond(A))
    # plt.spy(A)
    # plt.show()

    plot_figure_1d(x, u, u_exact)

    return u


# diffusion_solver_1d(p, xl, xr, nelem, quad_type, flux_type='BR1', nrefine, boundary_type=None, b=1, n=1):
u = poisson_1d(2, 0, 2*np.pi, 5, 'LGL', 'BR2', 4, 'nPeriodic', 'sbp_sat', a=0, b=1, n=19, app=1)

