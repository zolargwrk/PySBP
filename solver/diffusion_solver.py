import numpy as np
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator2D
from solver.plot_figure import plot_figure_1d, plot_figure_2d


def diffusion_solver_1d(p, xl, xr, nelem, t0, tf, quad_type, flux_type='BR1', boundary_type=None, a=1, n=1):

    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n)

    # refine mesh uniformly
    nrefine = 0  # number of uniform refinements
    for i in range(0, nrefine):
        mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)
        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n)

    x = rhs_data['x']
    n = rhs_data['n']  # degrees of freedom

    # nx = MeshTools1D.normals_1d(nelem)

    x = x.reshape((n, nelem), order='F')
    u = np.sin(x)

    rhs_calculator = RHSCalculator.rhs_diffusion_1d
    self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, None, flux_type, boundary_type)
    u = TimeMarcher.low_storage_rk4_1d(self_time_marcher, 0.25, x, a)

    u_exact = np.exp(-tf)*np.sin(x)
    plot_figure_1d(x, u, u_exact)

    return u


# diffusion_solver_1d(p, xl, xr, nelem, t0, tf, quad_type, flux_type='BR1', boundary_type=None, a=1, n=1):
u = diffusion_solver_1d(2, 0, 2*np.pi, 20, 0, 0.8, 'LG', 'BR1', 'nPeriodic', a=1, n=10)