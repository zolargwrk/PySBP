import numpy as np
import pickle
import matplotlib.pyplot as plt
import anytree
import graphviz
from types import SimpleNamespace
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import MaxNLocator
from anytree.exporter import DotExporter
from solver.diffusion_solver import poisson_sbp_2d
from solver.problem_statements import advec_diff1D_problem_input
from solver.advection_diffusion_solver import advec_diff_1d


def save_results(h=0.8, nrefine=2, sbp_families=None, sats=None, ps=None, solve_adjoint=False, save_results=False,
                 calc_cond=False, calc_eigvals=False, dim=2, stencil=('wide', 'narrow'), imp=('trad', 'elem'),
                 prob=('Diff', 'AdvDiff', 'Adv'), n=25, showMesh=False, p_map=1, curve_mesh=False, plot_fig=False):

    # setup default values based on input
    sbp_families, sats, ps, degrees, stencil, prob = input_defualt(sbp_families, sats, ps, stencil, prob, dim)

    # create a data tree using anytree module
    all_results = make_data_tree(sbp_families, sats, degrees, dim, stencil, imp, prob)

    if dim == 2:
        # render tree (png file is saved in "visual" directory)
        DotExporter(all_results).to_dotfile('all_results_2d.dot')
        graphviz.Source.from_file('all_results_2d.dot')
        graphviz.render('dot', 'png', 'all_results_2d.dot')
    elif dim == 1:
        # render tree (png file is saved in "visual" directory)
        DotExporter(all_results).to_dotfile('all_results_1d.dot')
        graphviz.Source.from_file('all_results_1d.dot')
        graphviz.render('dot', 'png', 'all_results_1d.dot')

    if dim == 2:
        # solve problem and add result to tree leaves
        for f in range(0, len(sbp_families)):
            for s in range(0, len(sats)):
                for p in range(0, len(ps)):

                    # solve the Poisson problem and obtain data
                    soln = poisson_sbp_2d(ps[p], h, nrefine, sbp_families[f], sats[s], solve_adjoint, plot_fig=plot_fig,
                                          calc_condition_num=calc_cond, calc_eigvals=calc_eigvals, showMesh=showMesh,
                                          p_map=p_map, curve_mesh=curve_mesh)

                    # add data to the leaves of the tree
                    leaf = all_results.children[0].children[f].children[s].children[p]
                    anytree.Node('data', parent=leaf, results=soln)

        # save result
        if save_results:
            path='C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual' \
                 '\\poisson2d_results\\'
            if calc_eigvals:
                with open(path + 'results_poisson2D_eigvals.pickle', 'wb') as outfile:
                    pickle.dump(all_results, outfile)
            else:
                with open(path + 'results_poisson2D.pickle', 'wb') as outfile:
                    pickle.dump(all_results, outfile)

    elif dim == 1:
        # solve problem and add result to tree leaves
        for f in range(len(sbp_families)):
            for t in range(len(stencil)):  # wide or narrow stencil
                for i in range(len(imp)):  # traditional or element type implementation
                    for s in range(len(sats)):
                        for p in range(len(ps)):
                            for r in range(len(prob)):
                                # set parameter to solve the Poisson problem
                                a = 0
                                b = 1
                                # change if problem is different from Poisson
                                if prob[r] == 'AdvDiff':
                                    a = 1
                                    b = 1
                                elif prob[r] == 'Adv':
                                    a = 1
                                    b = 0

                                # solve the Poisson problem and obtain data
                                soln = advec_diff_1d(ps[p], 0, 1, 2, sbp_families[f], 'upwind', sats[s], nrefine,
                                                     imp[i], 'nPeriodic', 'sbp_sat', advec_diff1D_problem_input, a, b,
                                                     n, app=t+1)

                                # add data to the leaves of the tree
                                leaf = all_results.children[0].children[f].children[t].children[i].children[s].\
                                    children[p].children[r]
                                anytree.Node('data', parent=leaf, results=soln)

        # save result
        if save_results:
            path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual' \
                   '\\poisson1d_results\\'
            if calc_eigvals:
                with open(path + 'results_poisson1D_eigvals.pickle', 'wb') as outfile:
                    pickle.dump(all_results, outfile)
            else:
                with open(path + 'results_poisson1D.pickle', 'wb') as outfile:
                    pickle.dump(all_results, outfile)

    return all_results


def analyze_results_2d(sbp_families=None, sats=None, ps=None, plot_by_family=False, plot_by_sat=False,
                       plot_spectrum=False, plot_spectral_radius=False, plot_sparsity=False, plot_adj_by_family=False,
                       plot_adj_by_sat=False, run_results=None, save_fig=False):

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\temp4\\'
    if run_results is None:
        # solve and obtain results or open saved from file
        with open(path+'results_poisson2D.pickle', 'rb') as infile:
            all_results = pickle.load(infile)
    else:
        all_results = run_results
    dim = 2
    # setup default values based on input
    sbp_families, sats, ps, degrees, _, _ = input_defualt(sbp_families, sats, ps)

    # setup plot options
    pltsetup_dict = plot_setup(sbp_families, sats, dim)
    pltsetup = SimpleNamespace(**pltsetup_dict)

    # plot solution by sbp family, i.e., 1 family with varying SAT types
    if plot_by_family:
        for p in range(len(ps)):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):

                    # get results from saved tree file
                    res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - 3
                    end = len(r.hs)

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(1)
                    plt.loglog(r.hs, r.errs_soln, pltsetup.markers[s], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                                label='SBP-{} $|$ {} $|$ {} $|$r={:.2f}'.format(pltsetup.sbp_fam[sbp_families[f]],
                                pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in solution')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)

                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarOper_{}_p{}.pdf'.format(sbp_families[f],
                                                                                                ps[p]), format='pdf')

                    # plot functional convergence rates
                    plt.figure(2)
                    plt.loglog(r.hs, r.errs_func, pltsetup.markers[s], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                                label='SBP-{} $|${}$|${}$|$ r={:.2f}'.format(pltsetup.sbp_fam[sbp_families[f]],
                                pltsetup.sat_name[sats[s]], degrees[p], conv_func))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in functional')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + '\\func_conv_rates\\errs_func_VarOper_{}_p{}.pdf'.format(sbp_families[f],
                                                                                                ps[p]), format='pdf')
                plt.show()
                plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_by_sat:
        for p in range(0, len(ps)):
            for s in range(0, len(sats)):
                for f in range(0, len(sbp_families)):

                    # get results from saved tree file
                    res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - 3
                    end = len(r.hs)

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(3)
                    plt.loglog(r.hs, r.errs_soln, pltsetup.markers[f], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                                label='SBP-{}| {}| {}| r={:.2f}'.format(pltsetup.sbp_fam[sbp_families[f]],
                                pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in solution')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarSAT_{}_p{}.pdf'.format(sats[s], ps[p]),
                                                                                                        format='pdf')

                    # plot functional convergence rates
                    plt.figure(4)
                    plt.loglog(r.hs, r.errs_func, pltsetup.markers[f], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                                label='SBP-{}| {}| {}| r={:.2f}'.format(pltsetup.sbp_fam[sbp_families[f]],
                                pltsetup.sat_name[sats[s]], degrees[p], conv_func))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in functional')
                    plt.legend()

                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + 'func_conv_rates\\errs_func_VarSAT_{}_p{}.pdf'.format(sats[s], ps[p]),
                                                                                                        format='pdf')
                plt.show()
                plt.close()

    # plot adjoint by sbp family, i.e., 1 family with varying SAT types
    if plot_adj_by_family:
        for p in range(len(ps)):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):

                    # get results from saved tree file
                    res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - 3
                    end = len(r.hs)

                    # calculate adjoint convergence rates
                    conv_adj = np.abs(
                        np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_adj[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(1)
                    plt.loglog(r.hs, r.errs_adj, pltsetup.markers[s], linewidth=pltsetup.lw,
                               markersize=pltsetup.ms,
                               label='SBP-{} $|$ {} $|$ {} $|$r={:.2f}'.format(pltsetup.sbp_fam[sbp_families[f]],
                                                                               pltsetup.sat_name[sats[s]],
                                                                               degrees[p], conv_adj))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in adjoint')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)

                    if save_fig:
                        plt.savefig(path + '\\adj_conv_rates\\errs_adj_VarOper_{}_p{}.pdf'.format(sbp_families[f],
                                                                                                    ps[p]),
                                    format='pdf')

                plt.show()
                plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_adj_by_sat:
        for p in range(0, len(ps)):
            for s in range(0, len(sats)):
                for f in range(0, len(sbp_families)):

                    # get results from saved tree file
                    res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - 3
                    end = len(r.hs)

                    # calculate solution convergence rates
                    conv_adj = np.abs(
                        np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_adj[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(3)
                    plt.loglog(r.hs, r.errs_adj, pltsetup.markers[f], linewidth=pltsetup.lw,
                               markersize=pltsetup.ms,
                               label='SBP-{}| {}| {}| r={:.2f}'.format(pltsetup.sbp_fam[sbp_families[f]],
                                                                       pltsetup.sat_name[sats[s]], degrees[p],
                                                                       conv_adj))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in adjoint')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + '\\adj_conv_rates\\errs_adj_VarSAT_{}_p{}.pdf'.format(sats[s], ps[p]),
                                    format='pdf')
                plt.show()
                plt.close()

    # plot spectrum of the system matrix
    if plot_spectrum:
        if run_results is None:
            with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile2: # needs result with all eigenvalues
                all_results = pickle.load(infile2)
        else:
            all_results = run_results

        for p in range(len(ps)):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):

                    # get results from saved tree file
                    res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # get real and imaginary parts
                    refine = 0  # refinment level
                    X = [x.real for x in r.eig_vals][refine]
                    Y = [x.imag for x in r.eig_vals][refine]


                    # plot eigenvalue spectrum
                    plt.rcParams.update({'font.size': 22, 'axes.labelsize': 22, 'legend.fontsize': 22,
                                         'xtick.labelsize': 22, 'ytick.labelsize': 22})
                    plt.scatter(X, Y, s=120,
                                label='SBP- {} $|$ {} $|$  p{} $|$  $\lambda_L$= ${:.2f}$ $|$  $\lambda_S$ = ${:.2e}$'.
                                format(pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], ps[p],
                                np.max(X), np.min(X)))
                    plt.xlabel(r'$Re(\lambda)$')
                    plt.ylabel(r'$Im(\lambda)$')
                    plt.legend()
                    plt.xscale('symlog')

                if save_fig:
                    plt.savefig(path + 'spectrum\\'+'spectrum_{}_p{}.pdf'.format(sbp_families[f], ps[p]), format='pdf')
                plt.show()
                plt.close()

    # plot spectral radius
    if plot_spectral_radius:
        if run_results is None:
            with open(path + 'results_poisson2D.pickle', 'rb') as infile3:
                all_results = pickle.load(infile3)
        else:
            all_results = run_results

        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        rholist = []
        plist = []
        for f in range(len(sbp_families)):
            for refine in range(nrefine):
                for s in range(len(sats)):
                    for p in range(len(ps)):

                        # get results from saved tree file
                        res = all_results.children[0].children[f].children[s].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # calculate spectral radius
                        rho = np.max(np.abs(r.eig_vals[refine])) #eig_vals[0] is the eigenvalues with no grid refinement
                        nelem = r.nelems[refine]
                        rholist.append(rho)
                        plist.append(p+1)

                    # plot spectral radius
                    plt.rcParams.update({'font.size': 28, 'axes.labelsize': 28, 'legend.fontsize': 28,
                                         'xtick.labelsize': 28, 'ytick.labelsize': 28})
                    plt.semilogy(plist, rholist, pltsetup.markers[s], linewidth=pltsetup.lw*3/4,
                                 markersize=pltsetup.ms*4/3, label='SBP- {} $|$ {} $|$ $n_e$={}'.
                                 format(pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], nelem))
                    plt.xlabel(r'operator degree, $p$')
                    plt.ylabel(r'spectral radius, $\max(|{\lambda}|)$')
                    plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.legend()

                    plist=[]
                    rholist=[]

                if save_fig:
                    plt.savefig(path + 'spectral_radius\\' + 'spectral_radius_{}_{}.pdf'.format(sbp_families[f],
                                                                                                nelem), format='pdf')
                plt.show()
                plt.close()

    if plot_sparsity:
        if run_results is None:
            with open(path + 'results_poisson2D.pickle', 'rb') as infile4:
                all_results = pickle.load(infile4)
        else:
            all_results = run_results

        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        nnz_per_nelems = []
        plist =[]
        for f in range(len(sbp_families)):
            for refine in range(nrefine):
                for s in range(len(sats)):
                    for p in range(len(ps)):

                        # get results from saved tree file
                        res = all_results.children[0].children[f].children[s].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # calculate spectral radius
                        nnz_elems = np.max(np.abs(r.nnz_elems[refine])) #eig_vals[0]:eigenvalues with no grid refinement
                        nelem = r.nelems[refine]
                        nnz_per_nelems.append(nnz_elems/nelem)
                        plist.append(p+1)

                    # plot spectral radius
                    plt.rcParams.update({'font.size': 28, 'axes.labelsize': 28, 'legend.fontsize': 28,
                                         'xtick.labelsize': 28, 'ytick.labelsize': 28})
                    plt.plot(plist, nnz_per_nelems, pltsetup.markers[s], linewidth=pltsetup.lw*3/4,
                             markersize=pltsetup.ms*4/3, label='SBP- {} $|$ {} $|$ $n_e$={}'.
                             format(pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], nelem))
                    plt.xlabel(r'operator degree, $p$')
                    plt.ylabel(r'$nnz(A)/n_{elems})$')
                    plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.legend()

                    plist = []
                    nnz_per_nelems = []

                if save_fig:
                    plt.savefig(path + 'sparsity\\' + 'sparsity_{}_{}.pdf'.format(sbp_families[f], nelem), format='pdf')
                plt.show()
                plt.close()
    return


def analyze_results_1d(sbp_families=None, sats=None, ps=None, stencil=None, imp=None, prob=None, plot_by_family=False,
                       plot_by_sat=False, plot_spectrum=False, plot_spectral_radius=False, plot_sparsity=False,
                       run_results=None, save_fig=False):

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson1d_results\\'
    if run_results is None:
        # solve and obtain results or open saved from file
        with open(path+'results_poisson1D.pickle', 'rb') as infile:
            all_results = pickle.load(infile)
    else:
        all_results = run_results

    dim = 1
    # setup default values based on input
    sbp_families, sats, ps, degrees, stencil, prob = input_defualt(sbp_families, sats, ps, stencil, prob, dim)

    # setup plot options
    pltsetup_dict = plot_setup(sbp_families, sats, dim, stencil)
    pltsetup = SimpleNamespace(**pltsetup_dict)

    imp_app=[]  # traditional or element type refinement
    for j in range(len(imp)):
        if imp[j] == 'trad':
            imp_app.append(0)
        elif imp[j] == 'elem':
            imp_app.append(1)
    imp_app = sorted(imp_app)

    app=[]  # wide or narrow stencil application
    for j in range(len(stencil)):
        if imp[j] == 'wide':
            app.append(1)
        elif imp[j] == 'narrow':
            app.append(2)
    app = sorted(app)

    # plot solution by sbp family, i.e., 1 family with varying SAT types
    if plot_by_family:
        for pr in range(len(prob)):
            for i in range(len(imp)):   # traditional or element type refinement
                for p in range(len(ps)):
                    for f in range(len(sbp_families)):
                        for s in range(len(sats)):
                            for t in range(len(stencil)):   # wide or narrow stencil
                                # get results from saved tree file
                                res = all_results.children[0].children[f].children[app[t]].children[imp_app[i]].\
                                    children[s].children[ps[p]-1].children[pr].children[0].results
                                # calculate degrees of freedom and add it to res dictionary
                                res['dof'] = np.asarray([x*y for x, y in zip(res['nelems'], res['ns'])])
                                r = SimpleNamespace(**res)

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = len(r.dof) - 3
                                end = len(r.dof)

                                # calculate solution convergence rates
                                conv_soln = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs[begin:end]), 1)[0])

                                # calculate functional convergence rates
                                conv_func = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs_func[begin:end]), 1)[0])

                                # plot solution convergence rates
                                plt.figure(1)
                                plt.loglog(1/r.dof, r.errs, pltsetup.markers[2*s+t], linewidth=pltsetup.lw,
                                           markersize=pltsetup.ms, label='{}-{}-{} $|$ {} $|$ {} $|$r={:.2f}'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]], imp[i],
                                                  pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                                plt.xlabel(r'$1/dof$')
                                plt.ylabel(r'error in solution')
                                plt.legend()
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)

                                if save_fig:
                                    plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarOper_{}_p{}.pdf'.
                                                format(sbp_families[f], ps[p]), format='pdf')

                                # plot functional convergence rates
                                plt.figure(2)
                                plt.loglog(1/r.dof, r.errs_func, pltsetup.markers[2*s+t], linewidth=pltsetup.lw,
                                           markersize=pltsetup.ms, label='{}-{}-{} $|${}$|${}$|$ r={:.2f}'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                            pltsetup.stencil_shortname[stencil[t]], imp[i],
                                                  pltsetup.sat_name[sats[s]], degrees[p], conv_func))
                                plt.xlabel(r'$1/dof$')
                                plt.ylabel(r'error in functional')
                                plt.legend()
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                                if save_fig:
                                    plt.savefig(path + '\\func_conv_rates\\errs_func_VarOper_{}_p{}.pdf'.
                                                format(sbp_families[f], ps[p]), format='pdf')
                    plt.show()
                    plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_by_sat:
        for pr in range(len(prob)):
            for i in range(len(imp)):   # traditional or element type refinement
                for p in range(len(ps)):
                    for s in range(len(sats)):
                        for f in range(len(sbp_families)):
                            for t in range(len(stencil)):   # wide or narrow stencil
                                # get results from saved tree file
                                res = all_results.children[0].children[f].children[t].children[imp_app[i]].children[s].\
                                    children[ps[p]-1].children[pr].children[0].results
                                # calculate degrees of freedom and add it to res dictionary
                                res['dof'] = np.asarray([x*y for x, y in zip(res['nelems'], res['ns'])])
                                r = SimpleNamespace(**res)

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = len(r.dof) - 3
                                end = len(r.dof)

                                # calculate solution convergence rates
                                conv_soln = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs[begin:end]), 1)[0])

                                # calculate functional convergence rates
                                conv_func = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs_func[begin:end]), 1)[0])

                                # plot solution convergence rates
                                plt.figure(1)
                                plt.loglog(1/r.dof, r.errs, pltsetup.markers[2*f+t], linewidth=pltsetup.lw,
                                           markersize=pltsetup.ms, label='{}-{}-{} $|$ {} $|$ {} $|$r={:.2f}'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]], imp[i],
                                                  pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                                plt.xlabel(r'$1/dof$')
                                plt.ylabel(r'error in solution')
                                plt.legend()
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)

                                if save_fig:
                                    plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarSAT_{}_p{}.pdf'.
                                                format(sbp_families[f], ps[p]), format='pdf')

                                # plot functional convergence rates
                                plt.figure(2)
                                plt.loglog(1/r.dof, r.errs_func, pltsetup.markers[2*f+t], linewidth=pltsetup.lw,
                                           markersize=pltsetup.ms, label='{}-{}-{} $|${}$|${}$|$ r={:.2f}'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]], imp[i],
                                                  pltsetup.sat_name[sats[s]], degrees[p], conv_func))
                                plt.xlabel(r'$1/dof$')
                                plt.ylabel(r'error in functional')
                                plt.legend()
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                                plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                                if save_fig:
                                    plt.savefig(path + '\\func_conv_rates\\errs_func_VarSAT_{}_p{}.pdf'.
                                                format(sbp_families[f], ps[p]), format='pdf')
                    plt.show()
                    plt.close()

    # plot of condition number, sparsity, and eigenvalue spectrum can be done in a similar fashion as the 2D case

    return


def input_defualt(sbp_families=None, sats=None, ps=None, stencil=None, prob=None, dim=2):

    if dim == 2:
        if sbp_families is None:
            sbp_families = ['gamma', 'omega', 'diagE']
        if sats is None:
            sats = ['BR1', 'BR2', 'LDG', 'CDG', 'BO', 'CNG']

        # change to all to small or capital letters
        sbp_families = [x.lower() for x in sbp_families]
        sats = [x.upper() for x in sats]

    elif dim == 1:
        if sbp_families is None:
            sbp_families = ['CSBP', 'CSBP_Mattsson2004', 'HGTL']
        if sats is None:
            sats = ['BR1', 'BR2', 'LDG', 'BO', 'CNG']
        if stencil is None:
            stencil = ['wide', 'narrow']
        if prob is None:
            prob = ['Diff', 'AdvDiff', 'Adv']

    if ps is None:
        ps = [1, 2, 3, 4]
        degrees = ['p1', 'p2', 'p3', 'p4']
    else:
        degrees = []
        for d in range(0, len(ps)):
            degrees.append('p' + str(ps[d]))

    return sbp_families, sats, ps, degrees, stencil, prob


def plot_setup(sbp_families, sats, dim=2, stencil=None):
    # dictionary to hold names of SBP families
    sbp_fam = {}
    stencil_shortname = {}
    if dim == 2:
        # 2D families
        if 'gamma' in sbp_families:
            sbp_fam['gamma'] = '$\Gamma$'
        if 'omega' in sbp_families:
            sbp_fam['omega'] = '$\Omega$'
        if 'diage' in sbp_families:
            sbp_fam['diage'] = 'E'

        markers = ['--*g', '--sy', ':<r', '-.ob', '--dm', ':hc', '--X']

    elif dim == 1:
        # 1D families
        if 'CSBP' in sbp_families:
            sbp_fam['CSBP'] = 'CSBP1'
        if 'CSBP_Mattsson2004' in sbp_families:
            sbp_fam['CSBP_Mattsson2004'] = 'CSBP2'
        if 'HGTL' in sbp_families:
            sbp_fam['HGTL'] = 'HGTL1'

        if 'wide' in stencil:
            stencil_shortname['wide'] = 'W'
        if 'narrow' in stencil:
            stencil_shortname['narrow'] = 'N'

        markers = ['--*g', '-*g', '--sy', '-sy', ':<r', '-<r', '-.ob', '-ob', '--dm', '-dm', ':hc', '-hc', '--X','-X']

    # dictionary to hold names of SATs
    sat_name = {}
    if 'BO' in sats:
        sat_name['BO'] = "  BO$\;\;$ "
    if 'BR2' in sats:
        sat_name['BR2'] = " BR2 "
    if 'BR1' in sats:
        sat_name['BR1'] = " BR1 "
    if 'LDG' in sats:
        sat_name['LDG'] = " LDG "
    if 'CDG' in sats:
        sat_name['CDG'] = " CDG "
    if 'IP' in sats:
        sat_name['IP'] = "SIPG"
    if 'CNG' in sats:
        sat_name['CNG'] = " CNG "
    if 'NIPG' in sats:
        sat_name['NIPG'] = "NIPG"

    # set plot parameters
    params = {'axes.labelsize': 20,
              'legend.fontsize': 20,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
              'text.usetex': False,          # True works only if results are read from pickle saved file
              'font.family': 'serif',
              'figure.figsize': [12, 9]}
    plt.rcParams.update(params)
    lw = 3  # lineweight
    ms = 15  # markersize

    return {'sbp_fam': sbp_fam, 'sat_name': sat_name, 'markers': markers, 'params': params, 'lw': lw, 'ms': ms,
            'stencil_shortname': stencil_shortname}


def make_data_tree(sbp_families, sats, degrees, dim=2, stencil=('wide', 'narrow'), imp=('trad', 'elem'),
                   prob=('Diff', 'AdvDiff', 'Adv')):

    # create a data tree using anytree module
    all_results = anytree.Node('all_results')
    sbp_family = anytree.Node('sbp_family', parent=all_results)
    if dim == 2:
        for f in range(len(sbp_families)):  # 2D sbp families
            anytree.Node(sbp_families[f], parent=sbp_family)
            for s in range(len(sats)):  # sat types
                anytree.Node(sats[s], parent=sbp_family.children[f])
                for d in range(len(degrees)):   # operator degree
                    anytree.Node(degrees[d], parent=sbp_family.children[f].children[s])
    elif dim == 1:
        for f in range(len(sbp_families)):  # operator types (Del Rey's or Mattsson's)
            anytree.Node(sbp_families[f], parent=sbp_family)
            for t in range(len(stencil)):   # wide or narrow stencil
                anytree.Node(stencil[t], parent=sbp_family.children[f])
                for i in range(len(imp)):   # traditional or element type implementation
                    anytree.Node(imp[i], parent=sbp_family.children[f].children[t])
                    for s in range(len(sats)):  # sat types
                        anytree.Node(sats[s], parent=sbp_family.children[f].children[t].children[i])
                        for d in range(len(degrees)):   # operator degree
                            anytree.Node(degrees[d], parent=sbp_family.children[f].children[t].children[i].children[s])
                            for r in range(len(prob)):
                                anytree.Node(prob[r],
                                         parent=sbp_family.children[f].children[t].children[i].children[s].children[d])

    return all_results


# ================================================  2D-plots  ======================================================== #
# give parameters for 2D solver and analyzer
# fam = ['gamma', 'omega', 'diagE']
# sat = ['BR1', 'BR2', 'LDG', 'CDG', 'BO', 'CNG']
# p = [1, 2, 3, 4]
fam = ['gamma','diage','omega']
sat = ['BR1','BR2']
p = [2,3,4]
p_map = 1
adj = False
plt_fam = False
plt_sat = False
plt_adj_fam = False
plt_adj_sat = False
calc_eigs = True
plt_eig = False
plt_rho = False
plt_sparsity = False
calc_cond_num = False
save_figure = False
curve_mesh = True

plt_soln = False
showMesh = False

soln = None
soln = save_results(h=2, nrefine=1, sats=sat, sbp_families=fam, ps=p, solve_adjoint=adj, save_results=False,
                    calc_cond=calc_cond_num, calc_eigvals=calc_eigs, showMesh=showMesh, p_map=p_map, curve_mesh=curve_mesh,
                    plot_fig=plt_soln)
analyze_results_2d(sats=sat, sbp_families=fam, ps=p, plot_by_family=plt_fam, plot_by_sat=plt_sat, plot_spectrum=plt_eig,
                   plot_spectral_radius=plt_rho, plot_sparsity=plt_sparsity,  plot_adj_by_sat=plt_adj_sat,
                   plot_adj_by_family=plt_adj_fam, run_results=soln, save_fig=save_figure)
# ==================================================================================================================== #

# ===============================================   1D-plots  ======================================================== #
# give parameters for 1D solver and analyzer
opers = ['CSBP', 'CSBP_Mattsson2004', 'HGTL']
# opers = ['CSBP']
# sat = ['BR1', 'BR2', 'LDG', 'BO', 'CNG']
sat = ['BR1', 'BR2']
# p = [1, 2, 3, 4]
p = [3, 4]
# sten = ['wide', 'narrow']
sten = ['narrow']
degree = ['p1', 'p2', 'p3', 'p4']
app = ['wide', 'narrow']
# imp_type = ['trad', 'elem']
imp_type = ['elem']
prob_type = ['Diff']

adj = False
plt_fam = False
plt_sat = True
plt_eig = False
plt_rho = False
plt_sparsity = False
calc_eigs = False
save_figure = False

# soln = None
# soln = save_results(nrefine=6, sbp_families=None, sats=None, ps=None, solve_adjoint=False, save_results=False,
#                  calc_cond=False, calc_eigvals=False, dim=1, stencil= app, imp=imp_type, prob=prob_type, n=25)
# analyze_results_1d(sats=sat, sbp_families=opers, ps=p, stencil=sten, imp=imp_type, prob=prob_type, plot_by_family=plt_fam,
#                    plot_by_sat=plt_sat, plot_spectrum=plt_eig, plot_spectral_radius=plt_rho, plot_sparsity=plt_sparsity,
#                    run_results=soln, save_fig=save_figure)
# ==================================================================================================================== #

