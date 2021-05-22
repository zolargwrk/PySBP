import numpy as np
from scipy.sparse.linalg import spsolve
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator2D, MeshGenerator1D
from solver.plot_figure import plot_figure_1d, plot_figure_2d, plot_conv_fig
from src.error_conv import calc_err, calc_conv
from types import SimpleNamespace
from scipy import sparse
from scipy import optimize
import matplotlib.pyplot as plt
from visual.mesh_plot import MeshPlot
import time
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian


def unsteady_burgers_2d(p, h, t0, tf, nu=1.0, cfl=1.0, nrefine=2, sbp_family='diagE', advection_sat_type='splitform',
                        diffusion_sat_type='BR2', domain_type='notperiodic', p_map=1, curve_mesh=False, showMesh=False,
                        steady=False):
    # the rectangular domain
    bL = 0 #-0.5
    bR = 1 #2*np.pi #0.5
    bB = 0 #-0.5
    bT = 1 #2*np.pi #0.5
    area = (bR - bL) * (bT - bB)
    pmap = p+1 # instead of the p_map provided (comment out to use user defined p_map value)

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
    btype = []
    if domain_type.lower() == 'notperiodic':
        btype = ['d', 'd', 'd', 'd']
    elif domain_type.lower() == 'periodic':
        btype = ['-', '-', '-', '-']

    ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map=p_map, curve_mesh=curve_mesh, domain_type=domain_type)

    errs_soln = list()
    errs_inf = list()
    hs = list()
    nelems = list()
    nnodes_list = list()
    niter_list = list()
    time_elapsed_list = list()

    # refine mesh
    for refine in range(0, nrefine):
        if refine == 0:
            mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
        else:
            # mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT, True)
            mesh = MeshTools2D.hrefine_uniform_2d(ass_data, bL, bR, bB, bT)

        # update assembled data for 2D implementation
        ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map, curve_mesh=curve_mesh, domain_type=domain_type)
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
        etov= adata.etov

        # diffusion coefficient
        lxx = nu * np.ones(x.shape)
        lxy = 0 * np.ones(x.shape)
        lyx = 0 * np.ones(x.shape)
        lyy = nu * np.ones(x.shape)

        Lxx = lxx.T.tolist()
        Lxy = lxy.T.tolist()
        Lyx = lyx.T.tolist()
        Lyy = lyy.T.tolist()

        LxxB = [sparse.block_diag(Lxx[i]) for i in range(0, len(Lxx))]
        LxyB = [sparse.block_diag(Lxy[i]) for i in range(0, len(Lxy))]
        LyxB = [sparse.block_diag(Lyx[i]) for i in range(0, len(Lyx))]
        LyyB = [sparse.block_diag(Lyy[i]) for i in range(0, len(Lyy))]

        LB = np.block([[sparse.block_diag(LxxB), sparse.block_diag(LxyB)], [sparse.block_diag(LyxB), sparse.block_diag(LyyB)]])

        # get operators on physical elements
        phy_data = MeshTools2D.map_operators_to_phy_2d(p, nelem, adata.H, adata.Dr, adata.Ds, adata.Er, adata.Es,
                                                       adata.R1, adata.R2, adata.R3, adata.B1, adata.B2, adata.B3,
                                                       adata.rx, adata.ry, adata.sx, adata.sy, adata.jac,
                                                       adata.surf_jac, adata.nx, adata.ny)
        phy = SimpleNamespace(**phy_data)

        # ------------------ on [0, 1] x [0,1]
        # u_exact = lambda x, y, t: 0.5 - np.tanh((x+y - t)/(2*nu))
        # u_exact_final = u_exact(x, y, tf)
        # ------------------ on [0, 1] x [0,1]
        # u_exact = lambda x, y, t: 1/(1 + np.exp((x + y - t - 1)/(2*nu)))
        # u_exact_final = u_exact(x, y, tf)

        # ------------------
        # u_exact = lambda x, y, t: 0*x

        # initial condition
        # u = u_exact(x, y, t0)
        # -----------
        # u = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
        # u_exact_final = 0 * u
        # -----------
        # u = 1/2*np.sin(x+y)
        # u_exact_final = 0 * u
        # ------------ on [-1,1] x [-1,1] ------------
        # u = np.exp(-10*(x**2+y**2))
        # u_exact_final = 0*u

        # steady problem------------------ on [-1,1] x [-1,1]
        u_exact = lambda x, y, t: np.exp(-40*((x-0.5)**2+(y-0.5)**2))
        # source_term = lambda x, y: (40*nu - 400*nu*(x**2+y**2))*np.exp(-10*(x**2+y**2)) - 20*(x+y)*np.exp(-20*(x**2+y**2))
        source_term = lambda x, y: (-3040 * nu - 6400 * nu * (x ** 2 + y ** 2) + 6400*nu*(x+y)) * np.exp(-40 * (x**2 - x + 1/2 + y**2 - y)) \
                                   - 80 * (x + y - 1) * np.exp(-80 * (x**2 - x + 1/2 + y**2 - y))

        u_exact_final = u_exact(x, y, 0)
        u = u_exact(x, y, 0)

        uDL_fun = None
        uDB_fun = None
        uDR_fun = None
        uDT_fun = None
        uNL_fun = None
        uNB_fun = None
        uNR_fun = None
        uNT_fun = None
        if domain_type.lower() == 'notperiodic':
            uDL_fun = lambda x, y, t: u_exact(bL, y, t)
            uDB_fun = lambda x, y, t: u_exact(x, bB, t)
            uDR_fun = lambda x, y, t: u_exact(bR, y, t)
            uDT_fun = lambda x, y, t: u_exact(x, bT, t)

        # define function to count number iteration in the Newton-Krylov minimization
        def count_iter(a, b):
            count_iter.counter += 1
        count_iter.counter = 0
        niter = 0
        time_elapsed = 0

        rhs_calculator = RHSCalculator.rhs_unsteady_burgers_sbp_2d
        if not steady:
            u, _ = TimeMarcher.low_storage_rk4_sbp_2d(u, t0, tf, rhs_calculator, phy_data, ass_data, a=np.array([1,1]), LB=LB,
                                                      cfl=cfl, eqn='burgers_viscous', advection_sat_type=advection_sat_type,
                                                      diffusion_sat_type=diffusion_sat_type, domain_type=domain_type,
                                                      uDL_fun=uDL_fun, uDR_fun=uDR_fun, uDB_fun=uDB_fun, uDT_fun=uDT_fun,
                                                      uNL_fun=uNL_fun, uNR_fun=uNR_fun, uNB_fun=uNB_fun, uNT_fun=uNT_fun,
                                                      bL=bL, bR=bR, bB=bB, bT=bT, nu=nu)
        elif steady:
            F = lambda u0: rhs_calculator(u0, 0, adata.xf, adata.yf, phy.DxB, phy.DyB, phy.HB, phy.BB, phy.RB, phy.nxB,
                                         phy.nyB, phy.jacB, adata.etoe, adata.etof, adata.bgrpD, adata.bgrpN, LB,
                                         advection_sat_type, diffusion_sat_type, domain_type, uDL_fun, uDR_fun, uDB_fun,
                                         uDT_fun, uNL_fun, uNR_fun, uNB_fun, uNT_fun, bL, bR, bB, bT, nu) \
                           + source_term(x, y)

            # jac = BroydenFirst()
            # kjac = KrylovJacobian(inner_M=InverseJacobian(jac))
            time_start = time.time()
            u = optimize.newton_krylov(F, u, f_tol=1e-6, method='gmres', verbose=True, callback=count_iter)
            # u = optimize.newton(F, u, tol=1e-4, maxiter=100)
            time_elapsed = time.time() - time_start

            niter = count_iter.counter
            niter_list.append(niter)
            time_elapsed_list.append(time_elapsed)

        # get global norm matrix
        Hg = sparse.block_diag(phy.HB)

        # calculate error
        err_soln = np.sqrt((u - u_exact_final).flatten(order="F") @ Hg @ (u - u_exact_final).flatten(order="F"))
        errs_soln.append(err_soln)

        err_inf = np.max(np.abs(u - u_exact_final))
        errs_inf.append(err_inf)

        # get number of elements and calculate element size
        nelems.append(nelem)
        nnodes_list.append(nnodes)
        h = np.sqrt(area/nelem)
        hs.append(h)

        # plot solution
        plot_figure_2d(x, y, u - u_exact_final)
        # plot_figure_2d(x, y, u)
        # plot_figure_2d(x, y, u_exact_final)

        if showMesh==True:
            showFacetNodes = True
            showVolumeNodes = True
            MeshPlot.plot_mesh_2d(nelem, r, s, x, y, xf, yf, vx, vy, etov, p_map, Lx, Ly, showFacetNodes, showVolumeNodes,
                                  saveMeshPlot=False, curve_mesh=curve_mesh, sbp_family=sbp_family)

            # visualize result
        print("error_soln =", "{:.4e}".format(err_soln), "; error_inf =", "{:.4e}".format(err_inf), "; nelem =", nelem, "; h =", "{:.4f}".format(h),
              "; ", sbp_family, "; ", advection_sat_type, "; ", diffusion_sat_type, "; p =", p, "; min_Jac =",
              "{:.6f}".format(np.min((sparse.block_diag(phy.jacB).diagonal()))))

    if nrefine>=3:
        conv_rate = np.polyfit(np.log(hs), np.log(errs_soln), 1)[0]
        conv_rate_inf = np.polyfit(np.log(hs), np.log(errs_inf), 1)[0]
        print("conv_rate =", conv_rate, "conv_rate_inf =", conv_rate_inf)

    return {'nelems': nelems, 'hs': hs, 'errs_soln': errs_soln, 'nnodes': nnodes_list,
            'uh': u, 'u_exact': u_exact_final, 'x': x, 'y': y, 'niter': niter_list, 'time_elapsed': time_elapsed_list}


# unsteady_burgers_2d(4, 0.8, 0.0, 0.01, nu=0.1, cfl=5e-1, nrefine=2, sbp_family='omega', advection_sat_type='twopoint',
#                     diffusion_sat_type='BR2', domain_type='periodic', p_map=2, curve_mesh=True, showMesh=True,
#                     steady=True)