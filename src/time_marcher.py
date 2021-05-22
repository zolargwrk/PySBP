import numpy as np
import quadpy
from types import SimpleNamespace
from src.ref_elem import Ref2D_DG
from scipy import sparse


class TimeMarcher:

    def __init__(self, u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun=None, flux_type='Central',
                 boundary_type=None, sat_type=None, app=1):
        self.u = u      # initial solution
        self.t0 = t0    # initial time
        self.tf = tf    # final time
        self.rhs_calculator = rhs_calculator    # method to evaluate the residual at every time step (it's a function)
        self.rhs_data = rhs_data
        self.u_bndry_fun = u_bndry_fun    # initial condition: input as a method (function of a and t)
        self.flux_type = flux_type  # flux type, either upwind or central, input as a string
        self.boundary_type = boundary_type
        self.sat_type = sat_type
        self.app = app

    def low_storage_rk4_1d(self, cfl, x, xl, a=1, b=1, uD_left=None, uD_right=None, uN_left=None, uN_right=None):
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
        boundary_type = self.boundary_type

        n = u.shape[0]
        nelem = u.shape[1]

        x2 = np.array([np.abs(xl- x[0, :]), np.abs(x[1, :] - x[0, :])]).flatten()
        x2[np.abs(x2) < 1e-12] = 0.0
        xmin = np.min(x2[np.nonzero(x2)])
        dt = 1/2*(cfl / abs(a)) * xmin    # advection
        # dt = cfl*(xmin**2)     # diffusion
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
                # advection
                rhs = rhs_calculator(u, rdata.x, t_local, a, rdata.xl, rdata.xr, rdata.d_mat, rdata.vmapM, rdata.vmapP,
                                     rdata.mapI, rdata.mapO, rdata.vmapI, rdata.tl, rdata.tr, rdata.rx,
                                     rdata.lift, rdata.fscale, rdata.nx, u_bndry_fun, flux_type, boundary_type)

                # diffusion
                # rhs = rhs_calculator(u, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx, rdata.rx,
                #                      rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO, rdata.vmapI,
                #                      rdata.vmapO, flux_type, self.sat_inviscid, boundary_type, rdata.db_mat, rdata.d2_mat,
                #                      b, self.app, uD_left, uD_right, uN_left, uN_right)

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
        boundary_type = self.boundary_type

        # unpack rhs_data
        rdata = SimpleNamespace(**rhs_data)

        n = u.shape[0] # n = (p+1)*(p+2)/2
        nelem = u.shape[1]
        nfp = p+1

        # set time step
        dt, _ = set_dt_2D(p, rdata.r, rdata.s, x, y, cfl)
        # dt = 1e-2
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
                rhs = rhs_calculator(u, time_loc, x, y, rdata.fx, rdata.fy, ax, ay, rdata.Dr, rdata.Ds, rdata.vmapM, rdata.vmapP,
                                     rdata.bnodes, rdata.bnodesB, nelem, nfp, btype, rdata.lift, rdata.fscale, rdata.nx,
                                     rdata.ny, u_bndry_fun, flux_type, boundary_type)
                res = rk4a[j] * res + dt * rhs
                u = u + rk4b[j] * res
            t += dt

        return u

    @staticmethod
    def low_storage_rk4_sbp_2d(u, t0, tf, rhs_calculator, phy_data, assembly_data, a=np.array([1,1]), LB=np.eye(2), cfl=1,
                               eqn='advection', advection_sat_type='upwind', diffusion_sat_type='BR2',
                               domain_type='notperiodic', uDL_fun=None, uDR_fun=None, uDB_fun=None, uDT_fun=None,
                               uNL_fun=None, uNR_fun=None, uNB_fun=None, uNT_fun=None, bL=None, bR=None, bB=None, bT=None, nu=None):
        """Low Storage Explicit RK4 method
            Inputs: cfl - CFL number
                    a - wave speed"""
        adata = SimpleNamespace(**assembly_data)
        phy = SimpleNamespace(**phy_data)
        p = (phy.nxB[0][0]).shape[0]-1

        nnodes = u.shape[0]
        nelem = u.shape[1]

        if domain_type.lower()=='periodic':
            etoe = adata.etoe_periodic
            etof = adata.etof_periodic
        else:
            etoe = adata.etoe
            etof = adata.etof

        # set time step
        # calculate the smallest mesh size
        hk = 2 * phy.jacB
        hmin = np.min(hk[np.nonzero(hk)])

        # calculate minimum distance on the reference element
        dx_min = 2
        for i in range(adata.r.shape[0]):
            for j in range(adata.s.shape[1]):
                dx = np.sqrt((adata.r[i] - adata.r[j])**2 + (adata.s[i] - adata.s[j])**2)
                if (dx > 0 and dx < dx_min):
                    dx_min = dx

        dt = 1e-2
        if eqn.lower() in ['advection', 'burgers_inviscid']:
            dt = cfl * (dx_min * hmin)
        elif eqn.lower() in ['diffusion', 'burgers_viscous']:
            # dt = cfl * (dx_min * hmin**2)
            dt = cfl/(2*p + 1) * hmin**2
        # dt=1/100000
        # low storage rk4 coefficients
        rk4a, rk4b, rk4c = coeff_low_storage_rk4()

        t = t0
        res = np.zeros((nnodes, nelem))
        rhs = np.zeros((nnodes, nelem))
        A = None

        # energy at initial condition
        energy0 = u.reshape((-1, 1), order='F').T @ sparse.block_diag(phy.HB) @ u.reshape((-1, 1), order='F')

        # time loop
        while t < tf:
            if t + dt > tf:
                dt = tf - t

            for j in range(0, 5):
                time_loc = t + rk4c[j] * dt
                if eqn.lower() == 'advection':
                    rhs, A = rhs_calculator(u, time_loc, adata.xf, adata.yf, phy.DxB, phy.DyB, phy.HB, phy.BB, phy.RB, phy.nxB,
                                         phy.nyB, etoe, etof, adata.bgrpD, a, advection_sat_type,
                                         domain_type, uDL_fun, uDR_fun, uDB_fun, uDT_fun, bL, bR, bB, bT)

                if eqn.lower() in ['burgers_inviscid', 'burgers_viscous']:
                    rhs = rhs_calculator(u, time_loc, adata.xf, adata.yf, phy.DxB, phy.DyB, phy.HB, phy.BB, phy.RB, phy.nxB,
                                         phy.nyB, phy.jacB, etoe, etof, adata.bgrpD, adata.bgrpN, LB,
                                         advection_sat_type, diffusion_sat_type, domain_type, uDL_fun, uDR_fun, uDB_fun,
                                         uDT_fun, uNL_fun, uNR_fun, uNB_fun, uNT_fun, bL, bR, bB, bT, nu)

                res = rk4a[j] * res + dt * rhs
                u = u + rk4b[j] * res
            t += dt
            print(t)
            energy1 = u.reshape((-1, 1), order='F').T @ sparse.block_diag(phy.HB) @ u.reshape((-1, 1), order='F')
            energy_change = energy1/energy0 - 1
            print('energy_change =', energy_change[0][0])

        return u, A


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

    mask = Ref2D_DG.fmask_2d(r, s, x, y)
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
