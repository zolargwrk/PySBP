import numpy as np
from src.assembler import Assembler
from src.time_marcher import TimeMarcher
from src.rhs_calculator import RHSCalculator
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator1D, MeshGenerator2D
from solver.plot_figure import plot_figure_1d, plot_figure_2d, plot_conv_fig
from src.error_conv import calc_err, calc_conv
from types import SimpleNamespace
from scipy.sparse.linalg import spsolve
from scipy import sparse
from visual.mesh_plot import MeshPlot
import matplotlib.pyplot as plt


def advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type='Central', boundary_type=None, n=1):

    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a,  nelem, n)
    errs = list()
    dofs = list()
    nelems = list()

    # refine mesh uniformly
    nrefine = 5   # number of uniform refinements
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type)
        else:
            mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)

        nelem = mesh['nelem']       # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n)

        x = rhs_data['x']
        n = rhs_data['n']   # degrees of freedom per element
        dofs.append(n * nelem)
        nelems.append(nelem)

        # nx = MeshTools1D.normals_1d(nelem)

        x = x.reshape((n, nelem), order='F')
        u = -np.sin(10*np.pi*x)

        def u_bndry_fun(a, t, xl, xr):
            if a >= 0:
                u0 = np.sin(10*np.pi*t)
                # u0 = -np.sin(10*np.pi*(xl-a*t))
            else:
                u0 = -np.sin(10*np.pi*(xr-a*t))
            return u0

        rhs_calculator = RHSCalculator.rhs_advection_1d
        self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type, boundary_type)
        u = TimeMarcher.low_storage_rk4_1d(self_time_marcher, 0.1, x, a)

        u_exact = -np.sin(10*np.pi*(x - a * tf))

        # error calculation
        rx = rhs_data['rx']
        h_mat = rhs_data['h_mat']
        err = calc_err(u, u_exact, rx, h_mat)
        errs.append(err)

    hs = (xr-xl)/(np.asarray(nelems))
    conv = calc_conv(hs, errs, 1, 4)
    np.set_printoptions(precision=3, suppress=False)
    print(np.asarray(conv))
    print(np.asarray(errs))

    plot_conv_fig(hs, errs, 1, 4)
    plot_figure_1d(x, u, u_exact)

    return u


def advection_solver_2d(p, h, t0, tf, cfl=1, flux_type='Upwind', boundary_type=None):

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h, -1, 1, -1, 1)

    # obtain all data necessary for the residual (RHS) calculation
    self_assembler = Assembler(p)
    rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    # refine mesh
    nrefine = 0
    for i in range(0, nrefine):
        mesh = MeshTools2D.hrefine_uniform_2d(rhs_data)
        rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    x = rhs_data['x']
    y = rhs_data['y']

    # set initial condition and wave speed constants
    ax = 1
    ay = 1
    u = np.sin(np.pi * x) * np.sin(np.pi*y)

    def u_bndry_fun(x, y, ax, ay, t):
        ub = np.sin(np.pi*(x - ax*t)) * np.sin(np.pi*(y - ay*t))
        return ub

    # set type of boundary: [left, right, bottom, top]
    btype = ['d', '-', 'd', '-']

    rhs_calculator = RHSCalculator.rhs_advection_2d
    self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type, boundary_type)
    u = TimeMarcher.low_storage_rk4_2d(self_time_marcher, p, x, y, btype, ax, ay, cfl)

    u_exact = np.sin(np.pi * (x-ax*tf)) * np.sin(np.pi*(y-ay*tf))
    err = np.linalg.norm((u - u_exact), 2)
    print(err)
    plot_figure_2d(x, y, u)

    # #--------------------------
    # rhs_calculator = RHSCalculator.rhs_advection_steady_2d
    # self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, u_bndry_fun, flux_type, boundary_type)
    # u = TimeMarcher.low_storage_rk4_2d(self_time_marcher, p, x, y, btype, ax, ay, cfl)

    return


# advection_solver_1d(p, xl, xr, nelem, t0, tf, a, quad_type, flux_type = 'Central')
# u = advection_solver_1d(2, 0, 1, 2, 0, 2, 1, 'CSBP', 'Upwind', 'nPeriodic', n=17)

# advection_solver_2d(p, h, t0, tf, cfl=1, flux_type='Central', boundary_type=None)
# u = advection_solver_2d(2, 0.5, 0, 1, cfl=1, flux_type='Upwind', boundary_type='nPeriodic')


def advection_solver_sbp_2d_steady(p, n_edge0, nrefine=1, sbp_family='diagE', upwind=1, p_map=1, curve_mesh=False, domain_type='periodic'):
    dim = 2
    nface = dim + 1
    nfp = p + 1
    ns = int((p + 1) * (p + 2) / 2)
    # the rectangular domain
    bL = 0
    bR = 2
    bB = 0
    bT = 2
    if upwind:
        upwind = 1
        flux_type = "upwind"
    else:
        flux_type = "symmetric"

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(n_edge0, bL, bR, bB, bT)
    if domain_type.lower() != 'periodic':
        btype = ['d', 'd', 'd', 'd']
    else:
        btype = ['-', '-', '-', '-']

    # set exact solution
    u_exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    f = lambda x, y: (np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + np.pi * np.cos(np.pi * y) * np.sin(np.pi * x))

    # u_exact = lambda x, y: np.sin(np.pi/2 * x)
    # f = lambda x, y: np.pi/2 * np.cos(np.pi/2 * x)

    ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map=p_map, curve_mesh=curve_mesh, domain_type=domain_type)
    errs_soln = list()
    hs = list()
    nelems = list()
    nnodes_list = list()

    # refine mesh
    for refine in range(0, nrefine):
        if refine == 0:
            mesh = MeshGenerator2D.rectangle_mesh(n_edge0, bL, bR, bB, bT)
        else:
            # mesh = MeshGenerator2D.rectangle_mesh((refine+1)*n_edge0, bL, bR, bB, bT)
            mesh = MeshTools2D.hrefine_uniform_2d(ass_data, bL, bR, bB, bT)

        # update assembled data for 2D implementation
        ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map, curve_mesh=curve_mesh, domain_type=domain_type)
        adata = SimpleNamespace(**ass_data)

        # extract variables from adata
        x = adata.x
        y = adata.y
        nelem = adata.nelem
        nnodes = adata.nnodes
        xf = adata.xf
        yf = adata.yf

        if domain_type.lower() != 'periodic':
            etoe = adata.etoe
            etof = adata.etof
        else:
            etoe = adata.etoe_periodic
            etof = adata.etof_periodic
        # initialize solution vectors
        u = 0 * x

        # get operators on physical elements
        phy_data = MeshTools2D.map_operators_to_phy_2d(p, nelem, adata.H, adata.Dr, adata.Ds, adata.Er, adata.Es,
                                                       adata.R1, adata.R2, adata.R3, adata.B1, adata.B2, adata.B3,
                                                       adata.rx, adata.ry, adata.sx, adata.sy, adata.jac,
                                                       adata.surf_jac, adata.nx, adata.ny)
        phy = SimpleNamespace(**phy_data)


        # advection coefficients
        IB = np.block([np.eye(nfp)] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        LrB = IB
        LsB = IB

        # get the advection coefficient
        Ln1B  = phy.nxB[0] * LrB + phy.nyB[0] * LsB
        Ln2B  = phy.nxB[1] * LrB + phy.nyB[1] * LsB
        Ln3B  = phy.nxB[2] * LrB + phy.nyB[2] * LsB
        Ln = [Ln1B, Ln2B, Ln3B]

        # upwind SATs
        T1kkB = 1/2*(phy.BB[0] @ (Ln[0] - upwind*np.abs(Ln[0])))
        T2kkB = 1/2*(phy.BB[1] @ (Ln[1] - upwind*np.abs(Ln[1])))
        T3kkB = 1/2*(phy.BB[2] @ (Ln[2] - upwind*np.abs(Ln[2])))
        Tkk = [T1kkB, T2kkB, T3kkB]

        HB_inv = np.linalg.inv(phy.HB)

        # construct the diagonal of the system matrix
        A_diag = sparse.csr_matrix(sparse.block_diag(phy.DxB) + sparse.block_diag(phy.DyB))

        # construct the SAT matrix
        sI = sparse.lil_matrix((nnodes * nelem, nnodes * nelem), dtype=np.float64)

        for elem in range(0, nelem):
            for face in range(0, nface):
                if not any(np.array_equal(np.array([elem, face]), rowD) for rowD in adata.bgrpD):
                    sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += HB_inv[elem] \
                                                @ (phy.RB[face][elem].T @ Tkk[face][elem] @ phy.RB[face][elem])
        for elem in range(0, nelem):
            for face in range(0, nface):
                elem_nbr = etoe[elem, face]
                # SAT terms from neighboring elements -- i.e., the subtracted part in terms containing (uk - uv)
                if elem_nbr != elem:
                    nbr_face = etof[elem, face]
                    sI[elem*nnodes:(elem+1)*nnodes, elem_nbr*nnodes:(elem_nbr+1)*nnodes] += HB_inv[elem]\
                                            @ (-phy.RB[face][elem].T @ Tkk[face][elem] @ np.flipud(phy.RB[nbr_face][elem_nbr]))

        # construct SAT matrix that multiplies the Dirichlet boundary vector
        sD = sparse.lil_matrix((nelem * nnodes, nelem * nfp * nface), dtype=np.float64)
        uD = np.zeros((nfp * nface, nelem))
        if domain_type.lower() != "periodic":
            # Dirichlet boundary condition
            for i in range(0, len(adata.bgrpD)):
                elem = adata.bgrpD[i, 0]
                face = adata.bgrpD[i, 1]
                # add boundary SAT terms
                sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += HB_inv[elem] @ phy.RB[face][elem].T \
                                @ Tkk[face][elem] @ phy.RB[face][elem]

            for i in range(0, len(adata.bgrpD)):
                elem = adata.bgrpD[i, 0]
                face = adata.bgrpD[i, 1]
                sD[elem*nnodes:(elem+1)*nnodes, (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] -= HB_inv[elem] \
                                @ (phy.RB[face][elem].T @ Tkk[face][elem])

            # set boundary conditions
            uD = MeshTools2D.set_bndry_advec_sbp_2D(xf, yf, adata.bgrpD, bL, bR, bB, bT, uDL_fun=u_exact,
                                                    uDB_fun=u_exact, uDR_fun=u_exact, uDT_fun=u_exact)

        A_mat = sparse.csr_matrix(A_diag) - sI.tocsr()
        sD_mat = sD.tocsr()
        sDf = (sD_mat @ uD.flatten(order="F")).reshape(-1, 1)
        fh = f(x, y).reshape(-1, 1, order='F') + sDf
        u = (spsolve(A_mat, fh)).reshape((nnodes, nelem), order="F")

        # get number of elements and calculate element size
        nelems.append(nelem)
        nnodes_list.append(nnodes)
        #h = 1 / np.sqrt(nelem)
        h = np.sqrt(2)/np.sqrt(2*(((n_edge0-1)*(2**refine))**2))
        hs.append(h)

        # error calculation
        Hg = sparse.block_diag(phy.HB)
        err_soln = np.sqrt((u - u_exact(x, y)).flatten(order="F") @ Hg @ (u - u_exact(x, y)).flatten(order="F"))

        errs_soln.append(err_soln)

        if refine != 0:
            rate = np.log(errs_soln[refine]/errs_soln[refine-1])/np.log(hs[refine]/hs[refine-1])
        else:
            rate = 0
        # result
        print("error_soln =", "{:.4e}".format(err_soln), "; nelem =", nelem, "; h =", "{:.4f}".format(h), "; ",
              sbp_family, "; ", flux_type, "; p =", p, "; rate =", "{:.5f}".format(rate), "; min_Jac =",
              "{:.5f}".format(np.min((sparse.block_diag(phy.jacB).diagonal()))))

        showMesh = False
        if showMesh == True:
            showFacetNodes = False
            showVolumeNodes = False
            MeshPlot.plot_mesh_2d(nelem, adata.r, adata.s, x, y, xf, yf, adata.vx, adata.vy, adata.etov, p_map,
                                  adata.Lx, adata.Ly, showFacetNodes, showVolumeNodes,
                                  saveMeshPlot=False, curve_mesh=curve_mesh, sbp_family=sbp_family)


    plot_soln = 1
    plot_conv = 0
    if plot_soln:
        plot_figure_2d(x, y, u_exact(x, y))
        plot_figure_2d(x, y, u)
    if plot_conv:
        conv_start = 2
        conv_end = nrefine
        #hs = np.sqrt(2)/np.asarray(np.sqrt(nelems))
        conv = calc_conv(hs, errs_soln, conv_start, conv_end)
        #np.set_printoptions(precision=3, suppress=False)
        print("conv rate: ", np.asarray(conv))
        # print(np.asarray(errs_soln))

        # plot_conv_fig(hs[0:], errs_soln[0:], conv_start, conv_end)

    return u


u = advection_solver_sbp_2d_steady(4, 3, nrefine=4, sbp_family='diage', upwind=False, p_map=1, curve_mesh=False,
                                   domain_type="notperiodic")