import numpy as np
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator2D
from solver.plot_figure import plot_figure_1d, plot_figure_2d


def advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type='Central', n=1):

    self_assembler = Assembler(p, quad_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, nelem, n)

    # refine mesh uniformly
    nrefine = 2     # number of uniform refinements
    for i in range(0, nrefine):
        mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)
        nelem = mesh['nelem']       # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, nelem, n)

    x = rhs_data['x']
    n = rhs_data['n']   # degrees of freedom

    # nx = MeshTools1D.normals_1d(nelem)

    x = x.reshape((n, nelem), order='F')
    u = np.sin(x)

    def u_init(a, t):
        u0 = - np.sin(a*t)
        return u0

    rhs_calculator = RHSCalculator.rhs_advection_1d
    self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_init, flux_type)
    u = TimeMarcher.low_storage_rk4_1d(self_time_marcher, 0.75, x, a)

    u_exact = np.sin(x - a * tf)
    plot_figure_1d(x, u, u_exact)

    return u


def advection_solver_2d(p, h, t0, tf, flux_type='Central', cfl=1):

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h)

    # obtain all data necessary for the residual (RHS) calculation
    self_assembler = Assembler(p)
    rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    # refine mesh
    nrefine = 3
    for i in range(0, nrefine):
        mesh = MeshTools2D.hrefine_uniform_2d(rhs_data)
        rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    x = rhs_data['x']
    y = rhs_data['y']

    # set initial condition and wave speed constants
    ax = 1
    ay = 0.1
    u = np.sin(np.pi * x) * np.sin(np.pi*y)

    def u_bndry_fun(x, y, ax, ay, t):
        ub = np.sin(np.pi*(x - ax*t)) * np.sin(np.pi*(y - ay*t))
        return ub

    # set type of boundary: [left, right, bottom, top]
    btype = ['d', '-', 'd', '-']

    rhs_calculator = RHSCalculator.rhs_advection_2d
    self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type)
    u = TimeMarcher.low_storage_rk4_2d(self_time_marcher, p, x, y, btype, ax, ay, cfl)

    u_exact = np.sin(np.pi * (x-ax*tf)) * np.sin(np.pi*(y-ay*tf))
    plot_figure_2d(x, y, u)

    return


# advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type = 'Central')
u = advection_solver_1d(1, 0, 2, 1, 0, 1, 2*np.pi, 'LG', 'Upwind', n=5)

# u = advection_solver_2d(3, 0.75, 0, 1, flux_type='Upwind', cfl=1)
