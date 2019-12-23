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
from solver.problem_statements import poisson1D_problem_input


def heat_1d(p, xl, xr, nelem, t0, tf, quad_type, flux_type='BR1', nrefine=1, boundary_type=None,
                        sat_type='dg_sat', a=1, b=1, n=1, app=1):

    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)
    errs = list()
    dofs = list()
    nelems = list()

    # boundary conditions
    uD_left = 0  # Dirichlet boundary at the left boundary
    uD_right = 0  # Dirichlet boundary at the right boundary
    uN_left = None  # Neumann boundary at the left boundary
    uN_right = None  # Neumann boundary at the right boundary

    # refine mesh uniformly
    # nrefine = 3  # number of uniform refinements
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        else:
            mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)

        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)

        x = rhs_data['x']
        n = rhs_data['n']  # degrees of freedom
        dofs.append(n*nelem)
        nelems.append(nelem)

        # nx = MeshTools1D.normals_1d(nelem)

        x = x.reshape((n, nelem), order='F')
        u = np.sin(x)

        rhs_calculator = RHSCalculator.rhs_diffusion_1d
        self_time_marcher = TimeMarcher(u, t0, tf, rhs_calculator, rhs_data, None, flux_type, boundary_type,
                                        sat_type, app)
        u = TimeMarcher.low_storage_rk4_1d(self_time_marcher, 0.2, x, xl, a, b, uD_left, uD_right, uN_left, uN_right)

        u_exact = np.exp(-tf)*np.sin(x)

        # error calculation
        rx = rhs_data['rx']
        h_mat = rhs_data['h_mat']
        err = calc_err(u, u_exact, rx, h_mat)
        errs.append(err)

    plot_err = 1
    if plot_err == 1:
        conv_start = 1
        conv_end = nrefine
        hs = (xr - xl) / (np.asarray(nelems))
        conv = calc_conv(hs, errs, conv_start, conv_end)
        np.set_printoptions(precision=3, suppress=False)
        print(np.asarray(conv))
        print(np.asarray(errs))

        plot_conv_fig(hs, errs, conv_start, conv_end)

    plot_figure_1d(x, u, u_exact)

    return u


# heat_1d(p, xl, xr, nelem, t0, tf, quad_type, flux_type='BR1', nrefine, boundary_type=None, b=1, n=1):
# u = heat_1d(1, 0, 2*np.pi, 5, 0, 0.1, 'CSBP', 'BR1', 5, 'nPeriodic', 'sbp_sat', a=0, b=1, n=6, app=2)

# For Order-Matched and compatible operators (app=2) the HGT works with BR2 scheme. HGTL and CSBP do not work, Next time
#   -- check why for order-matched operator the HGTL and CSBP operators do not work
#   -- check why HGT does not work for BR1
# **--** implement the methods for the Poisson problem and see if the issues still exist


def poisson_1d(p, xl, xr, nelem, quad_type, flux_type='BR1', nrefine=1, refine_type=None, boundary_type=None, sat_type='dg_sat',
              poisson1D_problem_input=None,  a=1, n=1, app=1):

    # get problem statement (ps)
    prob_input = poisson1D_problem_input()
    ps = SimpleNamespace(**prob_input)
    outputs = ps.choose_output()
    choose_outs = SimpleNamespace(**outputs)

    mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, 1, app)
    b = ps.var_coef(mesh['x'])
    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)
    errs = list()
    errs_adj = list()
    errs_func = list()
    dofs = list()
    nelems = list()

    # refine mesh uniformly
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        else:
            if refine_type=='trad':
                mesh = MeshTools1D.trad_refine_uniform_1d(rhs_data, p, quad_type, ps.var_coef, app)
                b = ps.var_coef(mesh['x'])
                n = mesh['n']
            else:
                mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)
                b = (ps.var_coef(mesh['x']))

        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)

        # extract some information from rdata
        rdata = SimpleNamespace(**rhs_data)
        rx = rdata.rx
        h_mat = rdata.h_mat
        n = rdata.n
        x = (rdata.x).reshape((n, nelem), order='F')

        dofs.append(n*nelem)
        nelems.append(nelem)

        # solve primal problem
        if choose_outs.prob == 'primal' or choose_outs.prob == 'all':
            # enforce boundary conditions (bc)
            bndry_conds = ps.boundary_conditions(xl, xr)
            bc = SimpleNamespace(**bndry_conds)

            A, fB = RHSCalculator.rhs_poisson_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx,
                                                        rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                                        rdata.vmapI, rdata.vmapO, flux_type, sat_type, boundary_type, rdata.db_mat,
                                                        rdata.d2_mat, b, app, bc.uD_left, bc.uD_right, bc.uN_left, bc.uN_right)

            # specify source term and add terms from the SATs to the source term (fB)
            f = - ps.source_term(x) + fB
            f = f.reshape((n * nelem, 1), order='F')

            # solve the linear system and get exact solution
            # u = (sparse.linalg.gmres(A, f, tol=1e-11, restart=20)[0]).reshape((n*nelem, 1), order='F')
            u = (spsolve(A, f)).reshape((n*nelem, 1))
            u_exact = ps.exact_solution(x).reshape((n*nelem, 1), order='F')

            # plot solution
            if choose_outs.plot_sol == 1:
                plot_figure_1d(x, u, u_exact)

            # error calculation for solution
            err = calc_err(u, u_exact, rx, h_mat)
            errs.append(err)

            # calculate functional output and exact functional
            J = ps.calc_functional(u, h_mat, rx)
            J_exact = ps.exact_functional(xl, xr)
            err_func = np.abs(J - J_exact)
            errs_func.append(err_func)

        # solve adjoint problem
        if choose_outs.prob == 'adjoint' or choose_outs.prob == 'all':
            adj_bcs = ps.adjoint_bndry(xl, xr)
            adj_bc = SimpleNamespace(**adj_bcs)
            A, gB = RHSCalculator.rhs_poisson_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx,
                                                        rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                                        rdata.vmapI, rdata.vmapO, flux_type, sat_type, boundary_type, rdata.db_mat,
                                                        rdata.d2_mat, b, app, adj_bc.psiD_left, adj_bc.psiD_right, adj_bc.psiN_left, adj_bc.psiN_right)

            # adjoint source term plus terms from SAT at boundary
            g = - ps.adjoint_source_term(x) + gB
            psi_exact = ps.exact_adjoint(x)
            psi = (spsolve(A, g)).reshape((n * nelem, 1))

            # plot solution
            if choose_outs.plot_sol == 1:
                plot_figure_1d(x, psi, psi_exact)

            # error calculation
            err_adj = calc_err(psi, psi_exact, rx, h_mat)
            errs_adj.append(err_adj)

    # plot error
    if choose_outs.prob == 'primal':
        if choose_outs.plot_err == 1 or choose_outs.func_conv == 1:
            if refine_type == 'trad':
                hs = (xr - xl) / (np.asarray(dofs))
            else:
                hs = (xr - xl) / (np.asarray(nelems))

            if choose_outs.plot_err == 1:
                conv_start = 3
                conv_end = nrefine - 0
                conv = calc_conv(hs, errs, conv_start, conv_end)
                print(np.asarray(conv))
                print(np.asarray(errs))
                plot_conv_fig(hs, errs, conv_start, conv_end)
            if choose_outs.func_conv == 1:
                conv_start = 1
                conv_end = nrefine - 4
                conv_func = calc_conv(hs, errs_func, conv_start, conv_end)
                print(np.asarray(conv_func))
                print(np.asarray(errs_func))
                plot_conv_fig(hs, errs_func, conv_start, conv_end)
    elif choose_outs.prob == 'adjoint':
        if choose_outs.plot_err == 1:
            conv_start = 2
            conv_end = nrefine - 0
            if refine_type == 'trad':
                hs = (xr - xl) / (np.asarray(dofs))
            else:
                hs = (xr - xl) / (np.asarray(nelems))
            conv_adj = calc_conv(hs, errs_adj, conv_start, conv_end)
            print(np.asarray(conv_adj))
            print(np.asarray(errs_adj))
            plot_conv_fig(hs, errs_adj, conv_start, conv_end)

    if choose_outs.show_eig==1:
        cond_A = sparse.linalg.norm(A) * sparse.linalg.norm(sparse.linalg.inv(A.tocsc()))
        print("{:.2e}".format(cond_A))
        # LR_eig = sparse.linalg.eigs(A, 1, which='LR', return_eigenvectors=False)
        # print(LR_eig)

        # eigA,_ = np.linalg.eig(A.todense())
        # cond_A = np.linalg.cond(A.todense())
        # max_eigA = np.round(np.max(eigA), 2)
        # print("{:.2e}".format(cond_A))
        # print(max_eigA)

        # xeig = [x.real for x in eigA]
        # yeig = [x.imag for x in eigA]
        # plt.scatter(xeig, yeig, color='red')
        # plt.xlabel('Real')
        # plt.ylabel('Imaginary')
        # plt.title('Oper: {}, SAT: {}, |min.eig| = {}, max.eig = {}'.format(quad_type, flux_type, min_eigA, max_eigA))
        # plt.show()
    # scipy.io.savemat('C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\DG\\Nodal DG Heasteven\\nodal-dg\\Codes1.1\\Codes1D\\Amatrices_python\\Amat_LGLDense_BR1sbp_p2.mat', {'A': A})
    return

# diffusion_solver_1d(p, xl, xr, nelem, quad_type, flux_type='BR1', nrefine, refine_type, boundary_type=None, b=1, n=1):
u = poisson_1d(3, 0, 1, 2, 'CSBP_Mattsson2004', 'BR2', 9, 'ntrad', 'nPeriodic', 'sbp_sat', poisson1D_problem_input, a=0, n=16, app=2)


#
# def check_time():
#     poisson_1d(5, 0, 2 * np.pi, 5, 'LGL', 'IP', 4, 'nPeriodic', 'sbp_sat', a=0, b=1, n=19, app=2)
# profile.run('check_time()')


def poisson_2d(p, h, nrefine=1, flux_type='BR1'):

    nface = 3
    nfp = p+1
    n = int((p+1)*(p+2)/2)
    errs = list()
    nelems = list()

    # generate mesh
    mesh = MeshGenerator2D.rectangle_mesh(h)

    # obtain all data necessary for the residual (RHS) calculation
    self_assembler = Assembler(p)
    rhs_data = Assembler.assembler_2d(self_assembler, mesh)

    # refine mesh
    for i in range(0, nrefine):
        if i == 0:
            rhs_data = Assembler.assembler_2d(self_assembler, mesh)
        else:
            mesh = MeshTools2D.hrefine_uniform_2d(rhs_data)
            rhs_data = Assembler.assembler_2d(self_assembler, mesh)

        rdata = SimpleNamespace(**rhs_data)
        x = rdata.x
        y = rdata.y
        nelem = rdata.nelem

        # # set up initial condition
        # u = np.sin(np.pi * x) * np.sin(np.pi * y)
        u = 0*x

        # set type of boundary: [left, right, bottom, top]
        btype = ['d', 'd', 'd', 'd']
        bcmaps = MeshTools2D.bndry_list(btype, rdata.bnodes, rdata.bnodesB)
        bcmap = SimpleNamespace(**bcmaps)
        mapD = bcmap.mapD
        vmapD = bcmap.vmapD
        mapN = bcmap.mapN
        vmapN = bcmap.vmapN
        uD = np.zeros((nfp*nface, nelem))
        qN = np.zeros((nfp*nface, nelem))
        # set up boundary conditions
        uD.reshape((nfp*nface*nelem, 1), order='F')[mapD] = 0

        A, M = RHSCalculator.rhs_poisson_2d(p, u, x, y, rdata.r, rdata.s, rdata.Dr, rdata.Ds, rdata.lift, rdata.nx, rdata.ny,
                                            rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapM, rdata.mapP,
                                            mapD, vmapD, mapN, vmapN, nelem, nfp, rdata.surf_jac, rdata.jac, flux_type)
        f = - 2*(np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)

        u = spsolve(A, f.reshape((n * nelem, 1), order='F'))
        uu = u.reshape(x.shape, order='F')
        u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)

        # error calculation
        err = np.linalg.norm((uu-u_exact), 2)
        errs.append(err)
        nelems.append(nelem)


    plot_err = 0
    if plot_err == 1:
        conv_start = 1
        conv_end = nrefine
        hs = np.asarray(np.sqrt(nelems))
        conv = calc_conv(hs, errs, conv_start, conv_end)
        np.set_printoptions(precision=3, suppress=False)
        print(np.asarray(conv))
        print(np.asarray(errs))

        plot_conv_fig(hs[1:], errs[1:], conv_start, conv_end)

    plot_figure_2d(x, y, u)
    print(errs)
    # plot_figure_2d(x, y, u_exact)

    return

# poisson_2d(2, 0.25, 1)


