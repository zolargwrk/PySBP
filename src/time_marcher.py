import numpy as np
import quadpy
from types import SimpleNamespace
from src.ref_elem import Ref2D


class TimeMarcher:

    def __init__(self, u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type='Central'):
        self.u = u      # initial solution
        self.t0 = t0    # initial time
        self.tf = tf    # final time
        self.rhs_calculator = rhs_calculator    # method to evaluate the residual at every time step (it's a function)
        self.rhs_data = rhs_data
        self.u_bndry_fun = u_bndry_fun    # initial condition: input as a method (function of a and t)
        self.flux_type = flux_type  # flux type, either upwind or central, input as a string

    def low_storage_rk4_1d(self, cfl, x, a=1):
        """Low Storage Explicit RK4 method
            Inputs: cfl - CFL number
                    a - wave speed"""
        u = self.u
        t0 = self.t0
        tf = self.tf
        rhs_calculator = self.rhs_calculator
        rhs_data = self.rhs_data
        u_bndry_fun = self.u_bndry_fun
        flux_type = self.flux_type

        n = u.shape[0]
        nelem = u.shape[1]

        xmin = np.min(np.abs(x[0, :] - x[1, :]))
        dt = (cfl/a)*xmin
        nstep = int(np.ceil(tf/(0.5*dt)))
        dt = tf/nstep

        # low storage rk4 coefficients
        rk4a, rk4b, rk4c = coeff_low_storage_rk4()

        # unpack rhs_data
        rdata = SimpleNamespace(**rhs_data)

        t = t0
        res = np.zeros((n, nelem))
        # time loop
        for i in range(0, nstep):
            for j in range(0, 5):
                t_local = t + rk4c[j]*dt
                rhs = rhs_calculator(u, t_local, a, rdata.d_mat, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                     rdata.vmapI, rdata.tl, rdata.tr, rdata.rx, rdata.lift, rdata.fscale, rdata.nx,
                                     u_bndry_fun, flux_type)
                res = rk4a[j]*res + dt*rhs
                u = u + rk4b[j]*res
            t += dt

        return u

    def low_storage_rk4_2d(self, p, x, y, btype, ax=1, ay=1, cfl=1):
        """Low Storage Explicit RK4 method
            Inputs: cfl - CFL number
                    a - wave speed"""
        u = self.u
        t0 = self.t0
        tf = self.tf
        rhs_calculator = self.rhs_calculator
        rhs_data = self.rhs_data
        u_bndry_fun = self.u_bndry_fun
        flux_type = self.flux_type

        # unpack rhs_data
        rdata = SimpleNamespace(**rhs_data)

        n = u.shape[0] # n = (p+1)*(p+2)/2
        nelem = u.shape[1]
        nfp = p+1

        # set time step
        # dt, _ = set_dt_2D(p, rdata.r, rdata.s, x, y, cfl)
        dt = 1e-2
        # low storage rk4 coefficients
        rk4a, rk4b, rk4c = coeff_low_storage_rk4()

        t = t0
        res = np.zeros((n, nelem))
        # time loop
        while t < tf:
            if t+dt > tf:
                dt = tf - t

            for j in range(0, 5):
                time_loc = t + rk4c[j] * dt
                rhs = rhs_calculator(u, time_loc, x, y, ax, ay, rdata.Dr, rdata.Ds, rdata.vmapM, rdata.vmapP,
                                     rdata.bnodes, rdata.bnodesB, nelem, nfp, btype, rdata.lift, rdata.fscale, rdata.nx,
                                     rdata.ny, u_bndry_fun, flux_type)
                res = rk4a[j] * res + dt * rhs
                u = u + rk4b[j] * res
            t += dt

        return u


def coeff_low_storage_rk4():
    rk4a = np.array([0.0, - 567301805773.0 / 1357537059087.0, - 2404267990393.0 / 2016746695238.0,
                     - 3550918686646.0 / 2091501179385.0, - 1275806237668.0 / 842570457699.0], dtype=float)
    rk4b = np.array([1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
                     1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
                     2277821191437.0 / 14882151754819.0], dtype=float)
    rk4c = np.array([0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
                     2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0], dtype=float)
    return rk4a, rk4b, rk4c


def set_dt_2D(p, r, s, x, y, cfl):

    mask = Ref2D.fmask_2d(r, s, x, y)
    # get distance between vertices (length of edges)
    dist = mask['distance_vertices']

    # calculate semi-perimeter
    speri = dist.sum(0)/2

    # calculate area of inscribed circle and the time scale for each element (for steady problems)
    area = np.sqrt(speri*(speri - dist[0, :])*(speri - dist[1, :])*(speri - dist[2, :]))
    dtscale = area/speri

    # calculate the minimum time scale for stability (for unsteady problems)
    ref = quadpy.line_segment.gauss_lobatto(p+1)
    x_min = abs(ref.points[1] - ref.points[0])
    dt = cfl*(2/3 * x_min * dtscale.min())

    return dt, dtscale

