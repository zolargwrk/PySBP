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
    x = rhs_data['x']
    n = rhs_data['n']

    nx = MeshTools1D.normals_1d(nelem)

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


def advection_solver_2d(p, h, t0, tf):

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h)

    # obtain all data necessary for the residual (RHS) calculation
    self_assembler = Assembler(p)
    rhs_data = Assembler.assembler_2d(self_assembler, mesh)
    x = rhs_data['x']
    y = rhs_data['y']
    bnodes = rhs_data['bnodes']

    # set initial condition and wave speed constants
    ax = 1
    ay = 0.1
    u = np.sin(np.pi * x) * np.sin(np.pi*y)

    def u_bndry_fun(x, y, ax, ay, t):
        ub = np.sin(np.pi*(x - ax*t)) * np.sin(np.pi*(y - ay*t))
        return ub

    # set type of boundary: [left, right, bottom, top]
    btype = ['d', '-', 'd', '-']
    # u = MeshTools2D.set_bndry(u, x, y, 0, btype, bnodes, u_bndry)

    rhs_calculator = RHSCalculator.rhs_advection_2d
    self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type='Central')
    u = TimeMarcher.low_storage_rk4_2d(self_time_marcher, p, x, y, btype, ax, ay)

    u_exact = np.sin(np.pi * (x-ax*tf)) * np.sin(np.pi*(y-ay*tf))
    plot_figure_2d(x, y, u)

    return


# advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type = 'Central')
# u = advection_solver_1d(4, 0, 2, 4, 0, 5, 2*np.pi, 'LGL', 'Upwind', n=40)

u = advection_solver_2d(4, 0.25, 0, 1)
