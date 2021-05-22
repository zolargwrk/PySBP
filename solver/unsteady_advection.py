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
import matplotlib.pyplot as plt
from visual.mesh_plot import MeshPlot

def unsteady_advection_2d(p, h, t0, tf, cfl, nrefine, sbp_family='diagE', advection_sat_type='upwind',
                          domain_type='notperiodic', p_map=1, calc_eigvals=False, curve_mesh=False, showMesh=False):
    # the rectangular domain
    bL = 0
    bR = 1
    bB = 0
    bT = 1
    area = (bR - bL) * (bT - bB)

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
    btype = []
    if domain_type.lower() == 'notperiodic':
        btype = ['d', '-', 'd', '-']
    elif domain_type.lower() == 'periodic':
        btype = ['-', '-', '-', '-']

    ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map=p_map, curve_mesh=curve_mesh, domain_type=domain_type)

    errs_soln = list()
    hs = list()
    nelems = list()
    nnodes_list = list()
    eig_vals = list()
    nnz_elems = list()

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

        # get operators on physical elements
        phy_data = MeshTools2D.map_operators_to_phy_2d(p, nelem, adata.H, adata.Dr, adata.Ds, adata.Er, adata.Es,
                                                       adata.R1, adata.R2, adata.R3, adata.B1, adata.B2, adata.B3,
                                                       adata.rx, adata.ry, adata.sx, adata.sy, adata.jac,
                                                       adata.surf_jac, adata.nx, adata.ny)
        phy = SimpleNamespace(**phy_data)

        # get advection speed
        a = np.array([[1.0], [1.0]])

        w = 2 * np.pi
        u_exact = lambda x, y, t: np.sin(w / Lx * (x - a[0, 0] * t)) * np.sin(w / Ly * (y - a[1, 0] * t))
        u_exact_final = u_exact(x, y, tf)

        # initial condition
        u = u_exact(x, y, t0)

        # set type of boundary: [left, right, bottom, top]
        btype=[]
        if domain_type.lower() == 'notperiodic':
            btype = ['d', '-', 'd', '-']
        elif domain_type.lower() == 'periodic':
            btype = ['-', '-', '-', '-']

        uDL_fun = None
        uDB_fun = None
        if domain_type.lower() == 'notperiodic':
            uDL_fun = lambda x, y, t: np.sin(w/Lx * (-a[0, 0]*t)) * np.sin(w/Ly * (y - a[1, 0]*t))
            uDB_fun = lambda x, y, t: np.sin(w/Lx * (x - a[0, 0]*t)) * np.sin(w/Ly * (-a[1, 0]*t))


        rhs_calculator = RHSCalculator.rhs_advection_unsteady_sbp_2d
        u, A = TimeMarcher.low_storage_rk4_sbp_2d(u, t0, tf, rhs_calculator, phy_data, ass_data, a, cfl, eqn='advection',
                                               advection_sat_type=advection_sat_type, domain_type=domain_type,
                                               uDL_fun=uDL_fun, uDR_fun=None, uDB_fun=uDB_fun, uDT_fun=None, bL=bL, bR=bR,
                                               bB=bB, bT=bT)

        # get global norm matrix
        Hg = sparse.block_diag(phy.HB)

        # calculate error
        err_soln = np.sqrt((u - u_exact_final).flatten(order="F") @ Hg @ (u - u_exact_final).flatten(order="F"))
        errs_soln.append(err_soln)

        # get number of elements and calculate element size
        nelems.append(nelem)
        nnodes_list.append(nnodes)
        h = np.sqrt(area/nelem)
        hs.append(h)

        # plot solution
        # plot_figure_2d(x, y, u)
        # plot_figure_2d(x, y, u - u_exact_final)

        # calculate eigen value, condition number, and number of nonzero elements
        if calc_eigvals:
            # eig_val = 0
            eig_val = np.linalg.eigvals(A.toarray())
        else:
            # eig_val = sparse.linalg.eigs(A, which='LR')[0]
            eig_val = 0
        eig_vals.append(eig_val)

        # number of nonzero elements
        nnz_elem = (np.abs(A) > 1e-12).count_nonzero()
        nnz_elems.append(nnz_elem)

        if showMesh==True:
            showFacetNodes = True
            showVolumeNodes = True
            MeshPlot.plot_mesh_2d(nelem, r, s, x, y, xf, yf, vx, vy, etov, p_map, Lx, Ly, showFacetNodes, showVolumeNodes,
                                  saveMeshPlot=False, curve_mesh=curve_mesh, sbp_family=sbp_family)

            # visualize result
        print("error_soln =", "{:.4e}".format(err_soln), "; nelem =", nelem, "; h =", "{:.4f}".format(h),
              "; ", sbp_family, "; ", advection_sat_type, "; p =", p, "; min_Jac =",
              "{:.6f}".format(np.min((sparse.block_diag(phy.jacB).diagonal()))))

    if nrefine >= 3:
        conv_rate = np.abs(np.polyfit(np.log10(hs), np.log10(errs_soln), 1)[0])
        print("conv_rate =", conv_rate)

    return {'nelems': nelems, 'hs': hs, 'errs_soln': errs_soln, 'eig_vals': eig_vals, 'nnodes': nnodes_list,
            'uh': u, 'u_exact': u_exact_final, 'x': x, 'y': y}

# unsteady_advection_2d(2, 0.8, 0.0, 0.1, cfl=1e-0, nrefine=3, sbp_family='diage', advection_sat_type='upwind',
#                           domain_type='notperiodic', p_map=2, curve_mesh=False, showMesh=False)