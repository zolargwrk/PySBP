import numpy as np
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools
from solver.plot_figure import plot_figure_1d


def advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type = 'Central'):

    n = p+1

    self_assembler = Assembler(p, quad_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, nelem)
    x = rhs_data['x']

    nx = MeshTools.normals_1d(nelem)

    x = x.reshape((n, nelem), order='F')
    u = np.sin(x)

    def u_init(a, t):
        u0 = - np.sin(a*t)
        return u0

    rhs_calculator = RHSCalculator.rhs_advection_1d
    self_time_marcher = TimeMarcher(u, t0, tf, nx, rhs_calculator, rhs_data, u_init, flux_type)
    u = TimeMarcher.low_storage_rk4(self_time_marcher, 0.75, x, a)

    u_exact = np.sin(x - a * tf)
    plot_figure_1d(x, u, u_exact)

    return u


# advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type = 'Central')
u = advection_solver_1d(15, 0, 2, 3, 0, 10, 2*np.pi, 'LG', 'Upwind')

