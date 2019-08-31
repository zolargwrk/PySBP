import numpy as np
from solver.plot_figure import plot_figure_1d
import matplotlib.pyplot as plt


class TimeMarcher:

    def __init__(self, u, t0, tf, nx, rhs_calculator, rhs_data, u_init, flux_type):
        self.u = u      # initial solution
        self.t0 = t0    # initial time
        self.tf = tf    # final time
        self.nx = nx    # normals at the faces
        self.rhs_calculator = rhs_calculator    # method to evaluate the residual at every time step (it's a function)
        self.rhs_data = rhs_data
        self.u_init = u_init    # initial condition: input as a method (function of a and t)
        self.flux_type = flux_type  # flux type, either upwind or central, input as a string

    def low_storage_rk4(self, cfl, x, a=1):
        """Low Storage Explicit RK4 method
            Inputs: cfl - CFL number
                    a - wave speed"""
        u = self.u
        t0 = self.t0
        tf = self.tf
        nx = self.nx
        rhs_calculator = self.rhs_calculator
        rhs_data = self.rhs_data
        u_init = self.u_init
        flux_type = self.flux_type

        n = u.shape[0]
        nelem = u.shape[1]

        xmin = np.min(np.abs(x[0,:] - x[1,:]))
        dt = (cfl/a)*xmin
        nstep = int(np.ceil(tf/(0.5*dt)))
        dt = tf/nstep

        # low storage rk4 coefficients
        rk4a = np.array([0.0, - 567301805773.0 / 1357537059087.0, - 2404267990393.0 / 2016746695238.0,
                         - 3550918686646.0 / 2091501179385.0, - 1275806237668.0 / 842570457699.0], dtype=float)
        rk4b = np.array([1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
                         1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
                         2277821191437.0 / 14882151754819.0], dtype=float)
        rk4c = np.array([0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
                         2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0], dtype=float)

        d_mat = rhs_data['d_mat']
        lift = rhs_data['lift']
        rx = rhs_data['rx']
        fscale = rhs_data['fscale']
        vmapM = rhs_data['vmapM']
        vmapP = rhs_data['vmapP']
        vmapB = rhs_data['vmapB']
        mapI = rhs_data['mapI']
        mapO = rhs_data['mapO']
        vmapI = rhs_data['vmapI']
        vmapO = rhs_data['vmapO']
        jac = rhs_data['jac']

        t = t0
        res = np.zeros((n, nelem))
        # time loop
        for i in range(0, nstep):
            for j in range(0, 5):
                t_local = t + rk4c[j]*dt
                rhs = rhs_calculator(u, t_local, a, d_mat, vmapM, vmapP, mapI, mapO, vmapI,
                         rx, lift, fscale, nx, u_init, flux_type)
                res = rk4a[j]*res + dt*rhs
                u = u + rk4b[j]*res
            t += dt

            # exact solution
            # u_exact = np.sin(x - a * t)
            # plot_figure_1d(x, u, u_exact)
            # plt.pause(1)


        return u
