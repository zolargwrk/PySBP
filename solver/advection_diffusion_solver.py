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
from solver.problem_statements import advection1D_problem_input, poisson1D_problem_input, advec_diff1D_problem_input
from matplotlib import pyplot as plt
from matplotlib.ticker import Locator


def advection_1d_steady(p, xl, xr, nelem, quad_type, nrefine=1, refine_type=None, advection1D_problem_input=None,
                        flux_type='upwind', n=1):
    # get problem statement (degree)
    prob_input = advection1D_problem_input()
    ps = SimpleNamespace(**prob_input)
    outputs = ps.choose_output()
    choose_outs = SimpleNamespace(**outputs)

    mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type)
    a = ps.var_coef(n)
    self_assembler = Assembler(p, quad_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n)
    errs = list()
    errs_adj = list()
    errs_func = list()
    dofs = list()
    nelems = list()
    # refine mesh uniformly
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type)
        else:
            if refine_type == 'trad':
                mesh = MeshTools1D.trad_refine_uniform_1d(rhs_data, p, quad_type, ps.var_coef)
                a = ps.var_coef(mesh['x'])
                n = mesh['n']
            else:
                mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)
                a = (ps.var_coef(mesh['x']))

        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n)

        # extract some information from rdata
        rdata = SimpleNamespace(**rhs_data)
        rx = rdata.rx
        h_mat = rdata.h_mat
        n = rdata.n
        x = (rdata.x).reshape((n, nelem), order='F')

        dofs.append(n * nelem)
        nelems.append(nelem)

        # solve primal problem
        if choose_outs.prob == 'primal' or choose_outs.prob == 'all':
            # enforce boundary conditions (bc)
            bndry_conds = ps.boundary_conditions(xl, xr)
            bc = SimpleNamespace(**bndry_conds)

            A, fB = RHSCalculator.rhs_advection_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.tl, rdata.tr,
                                                        rdata.rx, a, bc.uD_left, bc.uD_right, flux_type)

            # specify source term and add terms from the SATs to the source term (fB)
            f = ps.source_term(x) + fB
            f = f.reshape((n * nelem, 1), order='F')

            # solve the linear system and get exact solution
            u = (spsolve(A, f)).reshape((n * nelem, 1))
            u_exact = ps.exact_solution(x).reshape((n * nelem, 1), order='F')

            # plot solution
            if choose_outs.plot_sol == 1:
                plot_figure_1d(x, u, u_exact)

            # error calculation for solution
            err = calc_err(u, u_exact, rx, h_mat)
            errs.append(err)

            # calculate functional output and exact functional
            g = (ps.adjoint_source_term(x)).reshape((n*nelem, 1), order='F')
            J = ps.calc_functional(u, g, h_mat, rx)
            J_exact = ps.exact_functional(xl, xr)
            err_func = np.abs(J - J_exact)
            errs_func.append(err_func)

        # solve adjoint problem
        if choose_outs.prob == 'adjoint' or choose_outs.prob == 'all':
            adj_bcs = ps.adjoint_bndry(xl, xr)
            adj_bc = SimpleNamespace(**adj_bcs)
            a_adj = -a
            A, gB = RHSCalculator.rhs_advection_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.tl, rdata.tr,
                                                        rdata.rx, a_adj, adj_bc.psiD_left, adj_bc.psiD_right, flux_type)

            # adjoint source term plus terms from SAT at boundary
            g = ps.adjoint_source_term(x) + gB
            g = g.reshape((n * nelem, 1), order='F')
            psi_exact = (ps.exact_adjoint(x, xl, xr)).reshape((n * nelem, 1), order='F')
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
                conv_start = 2
                conv_end = nrefine - 2
                conv_func = calc_conv(hs, errs_func, conv_start, conv_end)
                print(np.asarray(conv_func))
                print(np.asarray(errs_func))
                plot_conv_fig(hs, errs_func, conv_start, conv_end)
    elif choose_outs.prob == 'adjoint':
        if choose_outs.plot_err == 1:
            conv_start = 3
            conv_end = nrefine - 0
            if refine_type == 'trad':
                hs = (xr - xl) / (np.asarray(dofs))
            else:
                hs = (xr - xl) / (np.asarray(nelems))
            conv_adj = calc_conv(hs, errs_adj, conv_start, conv_end)
            print(np.asarray(conv_adj))
            print(np.asarray(errs_adj))
            plot_conv_fig(hs, errs_adj, conv_start, conv_end)

    if choose_outs.show_eig == 1:
        cond_A = sparse.linalg.norm(A) * sparse.linalg.norm(sparse.linalg.inv(A.tocsc()))
        print("{:.2e}".format(cond_A))
        # LR_eig = sparse.linalg.eigs(A, 1, which='LR', return_eigenvectors=False)
        # print(LR_eig)
        eigA, _ = np.linalg.eig(-A.toarray())
        max_eigA = np.round(np.max(eigA), 2)
        print(max_eigA)

    return

# advection_1d_steady(p, xl, xr, nelem, quad_type, nrefine=1, refine_type=None, advection1D_problem_input=None,  a=1, n=1)
# advection_1d_steady(2, 0, 1, 1, 'CSBP', 5, 'trad', advection1D_problem_input, flux_type='upwind', n=25)


def poisson_1d(p, xl, xr, nelem, quad_type, flux_type='BR1', nrefine=1, refine_type=None, boundary_type=None, sat_type='sbp_sat',
              poisson1D_problem_input=None,  a=1, n=1, app=1):

    # get problem statement (degree)
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
            f = ps.source_term(x) - fB
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
            g = (ps.adjoint_source_term(x)).reshape((n*nelem, 1), order='F')
            J = ps.calc_functional(u, g, h_mat, rx)
            J_exact = ps.exact_functional(xl, xr)
            err_func = np.abs(J - J_exact)
            errs_func.append(err_func)

        if choose_outs.show_eig == 1:
            cond_A = sparse.linalg.norm(A) * sparse.linalg.norm(sparse.linalg.inv(A.tocsc()))
            # print("{:.2e}".format(cond_A))
            # LR_eig = sparse.linalg.eigs(-A, 1, which='LR', return_eigenvectors=False)
            # print(LR_eig)
            # eigA, _ = np.linalg.eig(-A.toarray())
            # max_eigA = np.round(np.max(eigA), 2)
            # print(max_eigA)

        # solve adjoint problem
        if choose_outs.prob == 'adjoint' or choose_outs.prob == 'all':
            adj_bcs = ps.adjoint_bndry(xl, xr)
            adj_bc = SimpleNamespace(**adj_bcs)
            A, gB = RHSCalculator.rhs_poisson_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx,
                                                        rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                                        rdata.vmapI, rdata.vmapO, flux_type, sat_type, boundary_type, rdata.db_mat,
                                                        rdata.d2_mat, b, app, adj_bc.psiD_left, adj_bc.psiD_right, adj_bc.psiN_left, adj_bc.psiN_right)

            # adjoint source term plus terms from SAT at boundary
            g = ps.adjoint_source_term(x) - gB
            g = g.reshape((n*nelem, 1), order='F')
            psi = (spsolve(A, g)).reshape((n * nelem, 1))
            psi_exact = (ps.exact_adjoint(x)).reshape((n * nelem, 1), order='F')

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
                conv_start = 2
                conv_end = nrefine - 0
                conv_func = calc_conv(hs, errs_func, conv_start, conv_end)
                print(np.asarray(conv_func))
                print(np.asarray(errs_func))
                plot_conv_fig(hs, errs_func, conv_start, conv_end)
    elif choose_outs.prob == 'adjoint':
        if choose_outs.plot_err == 1:
            conv_start = 3
            conv_end = nrefine - 0
            if refine_type == 'trad':
                hs = (xr - xl) / (np.asarray(dofs))
            else:
                hs = (xr - xl) / (np.asarray(nelems))
            conv_adj = calc_conv(hs, errs_adj, conv_start, conv_end)
            print(np.asarray(conv_adj))
            print(np.asarray(errs_adj))
            plot_conv_fig(hs, errs_adj, conv_start, conv_end)

    return

# u = poisson_1d(1, 0, 1, 1, 'CSBP', 'BR2', 4, 'ntrad', 'nPeriodic', 'sbp_sat', poisson1D_problem_input, n=25, app=2)


def advec_diff_1d(p, xl, xr, nelem, quad_type, flux_type_inv = 'upwind', flux_type_vis='BR1', nrefine=1, refine_type=None,
                  boundary_type=None, sat_type='sbp_sat', advec_diff1D_problem_input=None, a=0, b=1, n=1, app=1):

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson1d_results\\test\\'
    # get problem statement (degree)
    prob_input = advec_diff1D_problem_input()
    ps = SimpleNamespace(**prob_input)
    outputs = ps.choose_output()
    choose_outs = SimpleNamespace(**outputs)

    mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, 1, app)
    b = ps.var_coef_vis(b)
    a = ps.var_coef_inv(a)
    self_assembler = Assembler(p, quad_type, boundary_type)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)
    errs = list()
    errs_adj = list()
    errs_func = list()
    dofs = list()
    ns = list()
    nelems = list()
    cond_num = list()
    errs_J = list()
    uhs = list()
    xs = list()
    u_exacts = list()

    # refine mesh uniformly
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        else:
            if refine_type=='trad':
                mesh = MeshTools1D.trad_refine_uniform_1d(rhs_data, p, quad_type, ps.var_coef_vis, app)
                b = ps.var_coef_vis(mesh['x'])
                n = mesh['n']
            else:
                mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)
                b = (ps.var_coef_vis(mesh['x']))

        nelem = mesh['nelem']  # update the number of elements
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a, nelem, n, b, app)

        # extract some information from rdata
        rdata = SimpleNamespace(**rhs_data)
        rx = rdata.rx
        h_mat = rdata.h_mat
        n = rdata.n
        x = (rdata.x).reshape((n, nelem), order='F')

        dofs.append(n*nelem)
        ns.append(n)
        nelems.append(nelem)

        # solve primal problem
        if choose_outs.prob == 'primal' or choose_outs.prob == 'all':
            # enforce boundary conditions (bc)
            bndry_conds = ps.boundary_conditions(xl, xr)
            bc = SimpleNamespace(**bndry_conds)
            eqn='primal'

            A_vis, fB_vis, TD_left, TD_right, m_mat = RHSCalculator.rhs_poisson_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx,
                                                        rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                                        rdata.vmapI, rdata.vmapO, flux_type_vis, sat_type, boundary_type, rdata.db_mat,
                                                        rdata.d2_mat, b, app, bc.uD_left, bc.uD_right, bc.uN_left, bc.uN_right, eqn)

            A_inv, fB_inv = RHSCalculator.rhs_advection_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.tl,
                                                                  rdata.tr, rdata.rx, a, bc.uD_left, bc.uD_right, flux_type_inv)

            # specify source term and add terms from the SATs to the source term (fB)
            f = ps.source_term(x) - fB_vis + fB_inv
            f = f.reshape((n * nelem, 1), order='F')

            # system matrix
            A = A_vis + A_inv
            # solve the linear system and get exact solution
            u = (spsolve(A, f)).reshape((n*nelem, 1))
            u_exact = ps.exact_solution(x).reshape((n*nelem, 1), order='F')
            uhs.append(u)
            xs.append(x)
            u_exacts.append(u_exact)

            # plot solution
            if choose_outs.plot_sol == 1:
                g = (ps.adjoint_source_term(x)).reshape((n * nelem, 1), order='F')
                psi_exact = (ps.exact_adjoint(x, xl, xr)).reshape((n * nelem, 1), order='F')
                if i in range(1, nrefine): #nrefine-1:
                    plt.plot(x.flatten(order='F'), u.flatten(order='F'), '-or', label=r'$u_h$')
                    plt.plot(x.flatten(order='F'), u_exact.flatten(order='F'), '-k',  label=r'$u_{exact}$')
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'primal solution')
                    plt.legend()
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    plt.title(r'{}-SAT, elems={}, nodes per elem ={}'.format(flux_type_vis, int(nelem), int(len(x.flatten(order='F'))/nelem)))
                    # plt.savefig(path + '\\soln_conv_rates\\solution_{}_nelem{}.pdf'.format(flux_type_vis, nelem), format='pdf', bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    plt.plot(x.flatten(order='F'), (g * u_exact).flatten(order='F'), '-b', label=r'$\cal{GU}$')
                    plt.plot(x.flatten(order='F'), (f * psi_exact).flatten(order='F'), '-r', label=r'$\psi \cal{F}$')
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'$y$')
                    plt.legend()
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    plt.title(r'Analytical integrands in the functionals')
                    # plt.savefig(path + '\\func_conv_rates\\integrands_exact.pdf', format='pdf', bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    expon_lst = [3, 3, 5, 7]
                    plt.plot(x.flatten(order='F'), ((u_exact - u)).flatten(order='F'), '-r')
                    plt.yscale('symlog')
                    plt.yscale('symlog', linthreshy=1e-14) #10**(-expon_lst[i-1]))
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'$u_{exact} - u_h$')
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    err_sum = np.sum(u_exact - u)
                    plt.title(r'{}-SAT, elems={}, nodes per elem ={}, err sum={:.2e}'.format(flux_type_vis, int(nelem), int(len(x.flatten(order='F')) / nelem), err_sum))
                    # plt.savefig(path + '\\soln_conv_rates\\solution_error_{}_nelem{}.pdf'.format(flux_type_vis,nelem), format='pdf', bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    expon_lst = [3, 4, 5, 7]
                    plt.plot(x.flatten(order='F'), ((g*u_exact - g*u)).flatten(order='F'), '-r')
                    plt.yscale('symlog')
                    plt.yscale('symlog', linthreshy= 1e-14) #10 ** (-expon_lst[i - 1]))
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'$ {\cal{GU}}- g \circ u_h$')
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    err_sum = np.sum(g*u_exact - g*u)
                    plt.title(r'{}-SAT, elems={}, nodes per elem ={}, err sum={:.2e}'.format(flux_type_vis, int(nelem), int(len(x.flatten(order='F')) / nelem), err_sum))
                    # plt.savefig(path + '\\func_conv_rates\\integrand_error_{}_nelem{}.pdf'.format(flux_type_vis, nelem), format='pdf', bbox_inches='tight')
                    plt.show()
                    plt.close()

            # error calculation for solution
            err = calc_err(u, u_exact, rx, h_mat)
            errs.append(err)

            nnz_elem = A.count_nonzero()

            # calculate functional output and exact functional
            g = (ps.adjoint_source_term(x)).reshape((n*nelem, 1), order='F')
            J = ps.calc_functional(u, g, h_mat, rx,  rdata.db_mat, rdata.tr, rdata.tl, TD_left, TD_right)
            J_exact = ps.exact_functional(xl, xr)
            err_func = np.abs(J - J_exact)
            errs_func.append(err_func)

            if choose_outs.show_eig == 1:
                cond_A = sparse.linalg.norm(A) * sparse.linalg.norm(sparse.linalg.inv(A.tocsc()))
                cond_num.append(cond_A)
                # print("{:.2e}".format(cond_A))
                # nnzA = np.count_nonzero(A.toarray())
                # print(nnzA)
                # LR_eig = sparse.linalg.eigs(-A, 1, which='LR', return_eigenvectors=False)
                # print(LR_eig)
                eigA,_ = np.linalg.eig(-A.toarray())
                max_eigA = np.abs(np.round(np.max(np.real(eigA)), 2))
                min_eigA = np.abs(np.round((np.min(np.real(eigA))), 2))
                # nlast = n*nelem-(nelem-1)
                # eig_exact = (nlast)**2*(-2+2*np.cos(nlast*np.pi/(nlast + 1)))
                # print("{:.2e}".format(min_eigA))
                # print("{:.2e}".format(max_eigA))
                # print(min_eigA)
                # print(max_eigA)

                # xeig = [x.real for x in eigA]
                # yeig = [x.imag for x in eigA]
                # plt.scatter(xeig, yeig, color='red')
                # plt.xlabel('Real')
                # plt.ylabel('Imaginary')
                # plt.title('Oper: {}, SAT: {}, |min.eig| = {}, max.eig = {}'.format(quad_type, flux_type_vis, min_eigA, max_eigA))
                # plt.show()
                # we look for eigenvalues of -A instead of A because we multiplied A by -1 in rhs_calculator: rhs_poisson_1d_steady

        # solve adjoint problem
        if choose_outs.prob == 'adjoint' or choose_outs.prob == 'all':
            adj_bcs = ps.adjoint_bndry(xl, xr)
            adj_bc = SimpleNamespace(**adj_bcs)
            a_adj = -a  # advection coefficient changed to -a for the adjoint problem
            eqn='adjoint'

            A_vis, gB_vis, _, _, m_mat = RHSCalculator.rhs_poisson_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.lift, rdata.tl, rdata.tr, rdata.nx,
                                                        rdata.rx, rdata.fscale, rdata.vmapM, rdata.vmapP, rdata.mapI, rdata.mapO,
                                                        rdata.vmapI, rdata.vmapO, flux_type_vis, sat_type, boundary_type, rdata.db_mat,
                                                        rdata.d2_mat, b, app, adj_bc.psiD_left, adj_bc.psiD_right, adj_bc.psiN_left, adj_bc.psiN_right, eqn)

            A_inv, gB_inv = RHSCalculator.rhs_advection_1d_steady(n, nelem, rdata.d_mat, rdata.h_mat, rdata.tl,
                                                                  rdata.tr, rdata.rx, a_adj, adj_bc.psiD_left, adj_bc.psiD_right, flux_type_inv)

            A = A_vis + A_inv


            # adjoint source term plus terms from SAT at boundary
            g = ps.adjoint_source_term(x) - gB_vis + gB_inv
            g = g.reshape((n*nelem, 1), order='F')
            psi = (spsolve(A, g)).reshape((n * nelem, 1))
            psi_exact = (ps.exact_adjoint(x, xl, xr)).reshape((n * nelem, 1), order='F')

            if choose_outs.plot_sol == 1:
                if i in range(1, nrefine):# or i==2: #nrefine-1:
                    plt.plot(x.flatten(order='F'), psi.flatten(order='F'), '-or', label=r'$\psi_h$')
                    plt.plot(x.flatten(order='F'), psi_exact.flatten(order='F'), '-k', label=r'$\psi_{exact}$')
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'adjoint solution')
                    plt.legend()
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    plt.title(r'{}-SAT, elems={}, nodes per elem ={}'.format(flux_type_vis, int(nelem), int(len(x.flatten(order='F')) / nelem)))
                    # plt.savefig(path + '\\adj_conv_rates\\adjoint_{}_nelem{}.pdf'.format(flux_type_vis,nelem), format='pdf', bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    plt.plot(x.flatten(order='F'), ((psi_exact - psi)).flatten(order='F'), '-r')
                    plt.yscale('symlog')
                    plt.yscale('symlog', linthreshy=1e-14)
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'$\psi_{exact} - \psi_h$')
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    plt.title(r'{}-SAT, elems={}, nodes per elem ={}'.format(flux_type_vis, int(nelem), int(len(x.flatten(order='F')) / nelem)))
                    # plt.savefig(path + '\\adj_conv_rates\\adjoint_error_{}_nelem{}.pdf'.format(flux_type_vis,nelem), format='pdf', bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    plt.plot(x.flatten(order='F'), (g * u).flatten(order='F'), '-b', label=r'$g \circ u_h$')
                    plt.plot(x.flatten(order='F'), (f * psi).flatten(order='F'), '-r', label=r'$\psi_h \circ f$')
                    plt.xlabel(r'$x$')
                    plt.ylabel(r'$y$')
                    plt.legend()
                    params = {'text.usetex': True}
                    plt.rcParams.update(params)
                    plt.title(r'Numerical integrands in the functionals - {} SAT, nelem={}'.format(flux_type_vis, nelem))
                    # plt.savefig(path + '\\func_conv_rates\\integrands_numerical_{}_{}.pdf'.format(flux_type_vis, nelem), format='pdf', bbox_inches='tight')
                    # plt.show()
                    plt.close()

            err_adj = calc_err(psi, psi_exact, rx, h_mat)
            errs_adj.append(err_adj)

            # calculate functional using adjoint
            f = (ps.source_term(x)).reshape((n * nelem, 1), order='F')
            J_psi = ps.calc_functional(psi, f, h_mat, rx, rdata.db_mat, rdata.tr, rdata.tl, TD_left, TD_right, adj=True)
            err_J = np.abs(J_psi - J)
            errs_J.append(err_J)

            # M = sparse.block_diag([m_mat]*nelem)
            # kk = (u_exact.T @ (M - M.T) @ psi_exact)[0][0]
            # print("{:.2e}".format(kk))

        print("error_soln =", "{:.2e}".format(err), "; error_func =", "{:.2e}".format(err_func), "; nelem =", nelem,
              "; ", quad_type, "; ", flux_type_vis, "; p =", p, "; nnz_elem =", nnz_elem)

        # print("err_J =", "{:.2e}".format(err_J),"error_soln =", "{:.2e}".format(err), "error_adj =", "{:.2e}".format(err_adj), "; error_func =", "{:.2e}".format(err_func), "; nelem =", nelem,
        #       "; ", quad_type, "; ", flux_type_vis, "; p =", p, "; nnz_elem =", nnz_elem)

    if choose_outs.plot_sol == 1:
        plt.loglog(1/np.asarray(dofs), np.asarray(errs_J), ':ob')
        plt.xlabel(r'$1/dof$')
        plt.ylabel(r'$|I_h(u_h) - I_h(\psi_h)|$')
        params = {'text.usetex': True}
        plt.rcParams.update(params)
        plt.title(r'{}-SAT, numerical verification of $I_h(u_h) = I_h(\psi_h)$'.format(flux_type_vis))
        # plt.savefig(path + '\\func_conv_rates\\func_verify_{}.pdf'.format(flux_type_vis), format='pdf', bbox_inches='tight')
        # plt.show()
        plt.close()

    # plot error
    if choose_outs.prob == 'primal' or choose_outs.prob == 'all':
        if choose_outs.plot_err == 1 or choose_outs.func_conv == 1:
            if refine_type == 'trad':
                hs = (xr - xl) / (np.asarray(dofs))
            else:
                hs = (xr - xl) / (np.asarray(nelems))
                # hs = (xr - xl) / (np.asarray(dofs))

            if choose_outs.plot_err == 1:
                conv_start = 2
                conv_end = nrefine - 0
                conv = calc_conv(hs, errs, conv_start, conv_end)
                # print(np.asarray(conv))
                # print(np.asarray(errs))
                # plot_conv_fig(hs, errs, conv_start, conv_end)
            if choose_outs.func_conv == 1:
                conv_start = 2
                conv_end = nrefine - 0
                conv_func = calc_conv(hs, errs_func, conv_start, conv_end)
                # print(np.asarray(conv_func))
                # print(np.asarray(errs_func))
                # plot_conv_fig(hs, errs_func, conv_start, conv_end)

    if choose_outs.prob == 'adjoint' or choose_outs.prob == 'all':
        if choose_outs.plot_err == 1:
            conv_start = 2
            conv_end = nrefine - 0
            if refine_type == 'trad':
                hs = (xr - xl) / (np.asarray(dofs))
            else:
                hs = (xr - xl) / (np.asarray(nelems))
            conv_adj = calc_conv(hs, errs_adj, conv_start, conv_end)
            # print(np.asarray(conv_adj))
            # print(np.asarray(errs_adj))
            # plot_conv_fig(hs, errs_adj, conv_start, conv_end)

    return {'p': p, 'b': b, 'a': a, 'nelems': nelems, 'ns': ns, 'quad_type': quad_type, 'flux_type_vis': flux_type_vis,
            'errs': errs, 'errs_func': errs_func, 'cond_num': cond_num, 'uh': u, 'x':x, 'errs_adj': errs_adj,
            'uhs':uhs, 'xs': xs, 'u_exacts': u_exacts}

#advec_diff_1d(p, xl, xr, nelem, quad_type, flux_type_inv = 'upwind', flux_type_vis='BR1', nrefine=1, refine_type=None,
#                  boundary_type=None, sat_inviscid='sbp_sat', advec_diff1D_problem_input=None, a=0, b=1, n=1, app=1):
#CSBP_Mattsson2004
# u = advec_diff_1d(3, 0, 1, 2, 'LGL', 'upwind', 'BR2', 6, 'ntrad', 'nPeriodic', 'sbp_sat', advec_diff1D_problem_input, n=28, app=2)
