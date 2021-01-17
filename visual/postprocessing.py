import numpy as np
import pickle
import matplotlib.pyplot as plt
import anytree
import graphviz
from types import SimpleNamespace
from matplotlib.ticker import StrMethodFormatter, LogLocator
from matplotlib.ticker import MaxNLocator
from anytree.exporter import DotExporter
from solver.diffusion_solver import poisson_sbp_2d
from solver.problem_statements import advec_diff1D_problem_input
from solver.advection_diffusion_solver import advec_diff_1d
from src.ref_elem import Ref2D_SBP
from mpltools import annotation
import pandas as pd
from solver.plot_figure import plot_figure_1d, plot_figure_2d, plot_conv_fig

# show more than 5 coulumns when printing output to screeen
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)


def save_results(h=0.8, nrefine=2, sbp_families=None, sats=None, ps=None, solve_adjoint=False, save_results=False,
                 calc_cond=False, calc_eigvals=False, dim=2, stencil=('wide', 'narrow'), imp=('trad', 'elem'),
                 prob=('Diff', 'AdvDiff', 'Adv'), n=25, showMesh=False, p_map=1, curve_mesh=False, plot_fig=False,
                 modify_saved=False):

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
                        # find nodes
                        if modify_saved:
                            # SAT and SBP family list in saved data
                            sat_saved_list = ['BR1', 'BR2', 'LDG', 'CDG', 'BO', 'CNG']
                            sbp_famliy_saved_list = ['gamma', 'omega', 'diage']
                            p_saved_list = [0, 1, 2, 3]
                            fmod = sbp_famliy_saved_list.index(sbp_families[f])
                            smod = sat_saved_list.index(sats[s])
                            pmod = p_saved_list.index(ps[p]-1)
                        if calc_eigvals:
                            # with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile:
                            #     saved_results = pickle.load(infile)
                            #     soln = saved_results.children[0].children[f].children[s].children[p].children[0].results
                            if modify_saved:
                                with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile:
                                    # modify saved results
                                    saved_results = pickle.load(infile)
                                    saved_results.children[0].children[fmod].children[smod].children[pmod].children[0].results.clear()
                                    saved_results.children[0].children[fmod].children[smod].children[pmod].children[0].results.update(soln)
                                with open(path + 'results_poisson2D_eigvals.pickle', 'wb') as outfile:
                                    pickle.dump(saved_results, outfile)
                            else:
                                with open(path + 'results_poisson2D_eigvals.pickle', 'wb') as outfile:
                                    pickle.dump(all_results, outfile)
                        else:
                            if modify_saved:
                                # with open(path + 'results_poisson2D.pickle', 'rb') as infile:
                                #     saved_results = pickle.load(infile)
                                #     soln = saved_results.children[0].children[f].children[s].children[p].children[0].results
                                with open(path + 'results_poisson2D.pickle', 'rb') as infile:
                                    # modify saved results
                                    saved_results = pickle.load(infile)
                                    saved_results.children[0].children[fmod].children[smod].children[pmod].children[0].results.clear()
                                    saved_results.children[0].children[fmod].children[smod].children[pmod].children[0].results.update(soln)
                                with open(path + 'results_poisson2D.pickle', 'wb') as outfile:
                                    pickle.dump(saved_results, outfile)
                            else:
                                with open(path + 'results_poisson2D.pickle', 'wb') as outfile:
                                    pickle.dump(all_results, outfile)

    elif dim == 1:
        app = []  # wide or narrow stencil application
        for j in range(len(stencil)):
            if sten[j] == 'wide':
                app.append(1)
            elif sten[j] == 'narrow':
                app.append(2)

        app = sorted(app)
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
                                if sbp_families[f] in ['LG', 'LGL']:
                                    nelem = 5
                                    app_correct = 1
                                elif sbp_families[f] in ['CSBP', 'CSBP_Mattsson2013', 'CSBP_Mattsson2004', 'HGTL', 'HGT']:
                                    nelem = 1
                                    if ps[p]==1:
                                        n = 10
                                    elif ps[p]==2:
                                        n = 15
                                    elif ps[p]==3:
                                        n = 20
                                    elif ps[p]==4:
                                        n = 25

                                    app_correct = app[t]
                                else:
                                    nelem = 2
                                    app_correct = app[t]

                                soln = advec_diff_1d(ps[p], 0, 1, nelem, sbp_families[f], 'upwind', sats[s], nrefine,
                                                     imp[i], 'nPeriodic', 'sbp_sat', advec_diff1D_problem_input, a, b,
                                                     n, app_correct)

                                # add data to the leaves of the tree
                                leaf = all_results.children[0].children[f].children[t].children[i].children[s].\
                                    children[p].children[r]
                                anytree.Node('data', parent=leaf, results=soln)

        # save result
        if save_results:
            path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual' \
                   '\\poisson1d_results\\'
            if calc_eigvals:
                with open(path + 'results_poisson1D_eigvals_test.pickle', 'wb') as outfile:
                    pickle.dump(all_results, outfile)
            else:
                with open(path + 'results_poisson1D_test.pickle', 'wb') as outfile:
                    pickle.dump(all_results, outfile)

    return all_results


def analyze_results_2d(sbp_families=None, sats=None, ps=None, plot_by_family=False, plot_by_sat=False, plot_by_sat_all=False,
                       plot_spectrum=False, plot_spectral_radius=False, plot_sparsity=False, plot_adj_by_family=False,
                       plot_adj_by_sat=False, tabulate_cond_num=False, tabulate_density = False, tabulate_nnz=False,
                       run_results=None, save_fig=False):

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\'
    if run_results is None:
        # solve and obtain results or open saved from file
        with open(path+'results_poisson2D.pickle', 'rb') as infile:
            all_results = pickle.load(infile)

        pmin = np.min(ps) - 1
        pmax = np.max(ps)
    else:
        all_results = run_results

        pmin = 0
        pmax = len(ps)

    dim = 2
    # SAT and SBP family list in saved data
    sat_saved_list = ['BR1', 'BR2', 'LDG', 'CDG', 'BO', 'CNG']
    sbp_famliy_saved_list = ['gamma', 'omega', 'diage']
    p_saved_list = [0, 1, 2, 3]

    # setup default values based on input
    sbp_families, sats, ps, degrees, _, _ = input_defualt(sbp_families, sats, ps)

    # setup plot options
    pltsetup_dict = plot_setup(sbp_families, sats, dim)
    pltsetup = SimpleNamespace(**pltsetup_dict)

    # refinement level control
    b = 3
    e = 0
    if calc_eigs:
        nelem_all = np.array([14, 56, 224, 896])
    else:
        nelem_all = np.array([68, 272, 1088, 4352])
    
    hs = np.sqrt(2*20*10/nelem_all)

    # plot solution by sbp family, i.e., 1 family with varying SAT types
    if plot_by_family:
        for p in range(pmin, pmax):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # # get results from saved tree file
                    # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - b
                    end = len(r.hs) - e

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(1)
                    # label_fam_sol = 'SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(
                    #     pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], ps[p], conv_soln)
                    # label_fam_func = 'SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(
                    #     pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], ps[p], conv_func)
                    label_fam_sol='{}(${:.2f}$)'.format(pltsetup.sat_name[sats[s]], conv_soln)
                    label_fam_func='{}(${:.2f}$)'.format(pltsetup.sat_name[sats[s]], conv_func)
                    plt.loglog(r.hs, r.errs_soln, pltsetup.markers[s], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                               label=label_fam_sol)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'solution error')
                    plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))

                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarOper_{}_p{}.pdf'.format(sbp_families[f],
                                                                 p + 1), format='pdf', bbox_inches='tight')

                    # plot functional convergence rates
                    plt.figure(2)
                    plt.loglog(r.hs, r.errs_func, pltsetup.markers[s], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                               label=label_fam_func)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'functional error')
                    plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))
                    if save_fig:
                        plt.savefig(path + '\\func_conv_rates\\errs_func_VarOper_{}_p{}.pdf'.format(sbp_families[f],
                                                                              p + 1), format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_by_sat:
        for p in range(pmin, pmax):
            for s in range(0, len(sats)):
                for f in range(0, len(sbp_families)):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # # get results from saved tree file
                    # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - b
                    end = len(r.hs) - e

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(3)
                    # label_sat_sol='SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(pltsetup.sbp_fam[sbp_families[f]],
                    #             pltsetup.sat_name[sats[s]], ps[p], conv_soln)
                    # label_sat_func='SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(pltsetup.sbp_fam[sbp_families[f]],
                    #             pltsetup.sat_name[sats[s]], ps[p], conv_func)
                    label_sat_sol='SBP-{} (${:.2f}$)'.format(pltsetup.sbp_fam[sbp_families[f]], conv_soln)
                    label_sat_func = 'SBP-{} (${:.2f}$)'.format(pltsetup.sbp_fam[sbp_families[f]], conv_func)
                    plt.loglog(r.hs, r.errs_soln, pltsetup.markers[f], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                                label=label_sat_sol)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'solution error')
                    plt.legend(ncol=1, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))
                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarSAT_{}_p{}.pdf'.format(sats[s], p+1),
                                                                                 format='pdf', bbox_inches='tight')

                    # plot functional convergence rates
                    plt.figure(4)
                    plt.loglog(r.hs, r.errs_func, pltsetup.markers[f], linewidth=pltsetup.lw, markersize=pltsetup.ms,
                                label=label_sat_func)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'functional error')
                    plt.legend(ncol=1, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)

                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))
                    if save_fig:
                        plt.savefig(path + 'func_conv_rates\\errs_func_VarSAT_{}_p{}.pdf'.format(sats[s], p+1),
                                                                              format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_by_sat_all:
        for s in range(0, len(sats)):
            for p in range(pmin, pmax):
                for f in range(0, len(sbp_families)):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # # get results from saved tree file
                    # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - b
                    end = len(r.hs) - e

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(3)
                    # label_sat_sol='SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(pltsetup.sbp_fam[sbp_families[f]],
                    #             pltsetup.sat_name[sats[s]], p+1, conv_soln)
                    # label_sat_func='SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(pltsetup.sbp_fam[sbp_families[f]],
                    #             pltsetup.sat_name[sats[s]], p+1, conv_func)
                    label_sat_sol = 'SBP-{} $\;p={}$ (${:.2f}$)'.format(pltsetup.sbp_fam[sbp_families[f]], p+1, conv_soln)
                    label_sat_func = 'SBP-{} $\;p={}$ (${:.2f}$)'.format(pltsetup.sbp_fam[sbp_families[f]], p+1, conv_func)
                    plt.loglog(r.hs, r.errs_soln, pltsetup.markers_all[p][f], linewidth=pltsetup.lw-2, markersize=pltsetup.ms-3,
                                label=label_sat_sol)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'solution error')
                    plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)

                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))

                    # annotation.slope_marker((1.8 - ((7-p)/10)*(p+1), 1),
                    #                         slope=p + 6,
                    #                         size_frac=0.2,
                    #                         text_kwargs={'color': 'k', 'fontsize': 12})

                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarSAT_all_{}.pdf'.format(sats[s]),
                                    format='pdf', bbox_inches='tight')

                    # plot functional convergence rates
                    plt.figure(4)
                    plt.loglog(r.hs, r.errs_func, pltsetup.markers_all[p][f], linewidth=pltsetup.lw-2, markersize=pltsetup.ms-3,
                                label=label_sat_func)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'functional error')
                    plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)

                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))

                    if save_fig:
                        plt.savefig(path + 'func_conv_rates\\errs_func_VarSAT_all_{}.pdf'.format(sats[s]), format='pdf',
                                    bbox_inches='tight')
            plt.show()
            plt.close()

    # plot adjoint by sbp family, i.e., 1 family with varying SAT types
    if plot_adj_by_family:
        for p in range(pmin, pmax):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # # get results from saved tree file
                    # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - b
                    end = len(r.hs) - e

                    # calculate adjoint convergence rates
                    conv_adj = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_adj[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(1)
                    # label_fam_adj = 'SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(pltsetup.sbp_fam[sbp_families[f]],
                    #          pltsetup.sat_name[sats[s]], p + 1, conv_adj)
                    label_fam_adj = '{}(${:.2f}$)'.format(pltsetup.sat_name[sats[s]], conv_adj)
                    plt.loglog(r.hs, r.errs_adj, pltsetup.markers[s], linewidth=pltsetup.lw,
                               markersize=pltsetup.ms, label=label_fam_adj)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'adjoint error')
                    plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))

                    if save_fig:
                        plt.savefig(path + '\\adj_conv_rates\\errs_adj_VarOper_{}_p{}.pdf'.format(sbp_families[f], p+1),
                                    format='pdf', bbox_inches='tight')

                plt.show()
                plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_adj_by_sat:
        for p in range(pmin, pmax):
            for s in range(0, len(sats)):
                for f in range(0, len(sbp_families)):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)

                    # # get results from saved tree file
                    # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # set refinement levels where the convergence rates calculation begins and ends
                    begin = len(r.hs) - b
                    end = len(r.hs) - e

                    # calculate solution convergence rates
                    conv_adj = np.abs(
                        np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_adj[begin:end]), 1)[0])

                    # plot solution convergence rates
                    plt.figure(3)
                    # label_sat_adj='SBP-{} $\;$ {} $\;$ $p={}$ $\;$ $r={:.2f}$'.format(pltsetup.sbp_fam[sbp_families[f]],
                    #                                                    pltsetup.sat_name[sats[s]], p+1, conv_adj)
                    label_sat_adj='SBP-{} (${:.2f}$)'.format(pltsetup.sbp_fam[sbp_families[f]], conv_adj)
                    plt.loglog(r.hs, r.errs_adj, pltsetup.markers[f], linewidth=pltsetup.lw,
                               markersize=pltsetup.ms, label= label_sat_adj)
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'adjoint error')
                    plt.legend(labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=8, labelsize=22)
                    plt.gca().axes.tick_params(which='major', width=2, length=8, labelsize=22)
                    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 4, 6, 8]))

                    if save_fig:
                        plt.savefig(path + '\\adj_conv_rates\\errs_adj_VarSAT_{}_p{}.pdf'.format(sats[s], p+1),
                                    format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()

    # plot spectrum of the system matrix
    if plot_spectrum:
        if run_results is None:
            with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile2: # needs result with all eigenvalues
                all_results = pickle.load(infile2)
        else:
            all_results = run_results

        for p in range(pmin, pmax):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)
                    # # get results from saved tree file
                    # res = all_results.children[0].children[sbp_famliy_saved_list.index(sbp_families[f])].children[sat_saved_list.index(sats[s])].children[p].children[0].results
                    # # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # get real and imaginary parts
                    refine = 0  # refinment level
                    X = [x.real for x in r.eig_vals][refine]
                    Y = [x.imag for x in r.eig_vals][refine]


                    # plot eigenvalue spectrum
                    plt.rcParams.update({'font.size': 24, 'axes.labelsize': 26, 'legend.fontsize': 23,
                                         'xtick.labelsize': 26, 'ytick.labelsize': 26})
                    plt.rcParams['text.latex.preview'] = True
                    marker_spectrum = ['.', 'x', 'o', 's', 'd', '*']
                    marker_facecolor = ['r', 'b', 'none', 'none', 'none', 'none']
                    marker_edgecolor = ['r', 'b', 'k', 'g', 'c', 'm']

                    # label_spectrum='SBP- {} $\;$ {} $\;$  $p={}$ $\;$  $\lambda_L = {:.2f}$ $\;$  $\lambda_S = {:.2e}$'.\
                    #     format(pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], p+1, np.max(X), np.min(X))

                    label_spectrum = r'{} $\max(Re(\lambda))= {:.2f}$ $\;$  $\rho = {:.2e}$'. \
                        format(pltsetup.sat_name[sats[s]], np.max(X), np.max(np.abs(X)))

                    system_matrix = '_'
                    if sats[s] in {'BR1', 'BR2', 'LDG', 'CDG'}:
                        plt.scatter(X, Y+s, s=120, marker=marker_spectrum[s], facecolors=marker_facecolor[s], edgecolors=marker_edgecolor[s],
                                    label=label_spectrum)
                        plt.xlabel(r'$\lambda$')
                        system_matrix = 'symmetric'
                        plt.yticks(np.arange(4), ('0', '0', '0', '0'))
                        # plt.yscale('symlog')
                        # plt.xscale('symlog')

                        plt.legend(labelspacing=0.1, columnspacing=0.7, handletextpad=0.05, loc=(0, 0.065))
                        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


                    elif sats[s] in {'BO', 'CNG'}:
                        plt.scatter(X, Y, s=120, marker=marker_spectrum[s], facecolors=marker_facecolor[s],
                                    edgecolors=marker_edgecolor[s], label=label_spectrum)
                        plt.xlabel(r'$Re(\lambda)$')
                        plt.ylabel(r'$Im(\lambda)$')
                        system_matrix = 'asymmetric'
                        plt.legend(labelspacing=0.1, columnspacing=0.7, handletextpad=0.05, loc=(0, 0.065))
                        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                        plt.ticklabel_format(style='sci', axis='y', scilimits=(10, 0))

                    else:
                        plt.scatter(X, Y, s=120, marker=marker_spectrum[s], facecolors=marker_facecolor[s],
                                    edgecolors=marker_edgecolor[s],
                                    label=label_spectrum)

                        plt.legend(labelspacing=0.1, columnspacing=0.7, handletextpad=0.05, loc=(0, 0.065))
                        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

                if save_fig:
                        plt.savefig(path + 'spectrum\\'+'spectrum_{}_{}_p{}.pdf'.format(system_matrix, sbp_families[f], p+1),
                                    format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()

    # plot spectral radius
    if plot_spectral_radius:
        if run_results is None:
            with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile3:
                all_results = pickle.load(infile3)
        else:
            all_results = run_results

        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        rholist = []
        plist = []
        for f in range(len(sbp_families)):
            # for refine in range(nrefine):
            for s in range(len(sats)):
                for p in range(pmin, pmax):
                    if run_results is None:
                        # SAT and SBP family list in saved data
                        fmod = sbp_famliy_saved_list.index(sbp_families[f])
                        smod = sat_saved_list.index(sats[s])
                    else:
                        fmod = f
                        smod = s

                    # get results from saved tree file
                    res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                    r = SimpleNamespace(**res)
                    # # get results from saved tree file
                    # res = all_results.children[0].children[f].children[s].children[p].children[0].results
                    # r = SimpleNamespace(**res)

                    # calculate spectral radius
                    refine=0
                    rho = np.max(np.abs(r.eig_vals[refine])) #eig_vals[0] is the eigenvalues with no grid refinement
                    nelem = r.nelems[refine]
                    rholist.append(rho)
                    plist.append(p+1)

                # plot spectral radius
                plt.rcParams.update({'font.size': 28, 'axes.labelsize': 28, 'legend.fontsize': 28,
                                     'xtick.labelsize': 28, 'ytick.labelsize': 28})
                plt.semilogy(plist, rholist, pltsetup.markers[s], linewidth=pltsetup.lw*3/4,
                             markersize=pltsetup.ms*4/3, label='SBP- {} $|$ {} $|$ $n_e={}$'.
                             format(pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], nelem))
                plt.xlabel(r'operator degree, $p$')
                plt.ylabel(r'spectral radius, $\max(|{\lambda}|)$')
                plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.legend()

                plist=[]
                rholist=[]

            if save_fig:
                plt.savefig(path + 'spectral_radius\\' + 'spectral_radius_{}_{}.pdf'.format(sbp_families[f], nelem),
                            format='pdf', bbox_inches='tight')
            plt.show()
            plt.close()

    # Tabulate condition number
    if tabulate_cond_num:
        if run_results is None:
            with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile3:
                all_results = pickle.load(infile3)
        else:
            all_results = run_results

        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        cond_list = []
        row_list = []
        col_list = ['Degree', 'Operator']
        col_list.extend(sats)
        for refine in range(nrefine):
            for p in range(pmin, pmax):
                for f in range(len(sbp_families)):
                    row = [p+1, sbp_families[f]]
                    for s in range(len(sats)):
                        if run_results is None:
                            # SAT and SBP family list in saved data
                            fmod = sbp_famliy_saved_list.index(sbp_families[f])
                            smod = sat_saved_list.index(sats[s])
                        else:
                            fmod = f
                            smod = s

                        # get results from saved tree file
                        res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # get nnz of BR1 as a reference
                        resref = all_results.children[0].children[fmod].children[5].children[p].children[0].results
                        rref = SimpleNamespace(**resref)

                        # add condition number to row of the table
                        row.append(r.cond_nums[refine])

                        # uncomment the line below for relative spectral radius
                        # row.append(r.cond_nums[refine]/rref.cond_nums[refine])
                    row_list.append(row)
            cond_list.append(row_list)
            row_list = []

            df = pd.DataFrame(cond_list[refine], columns=col_list)
            pd.options.display.float_format = '{:.2e}'.format
            df.to_string(index=False)
            print(df, '\n')
            # print(df.to_latex(), '\n')

    if plt_cond:
        if run_results is None:
            with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile3:
                all_results = pickle.load(infile3)
        else:
            all_results = run_results

        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        nelem_list = []
        cond_num_list = []
        for p in range(pmin, pmax):
            for f in range(len(sbp_families)):
                for s in range(len(sats)):
                    for refine in range(nrefine):
                        # get results from saved tree file
                        res = all_results.children[0].children[f].children[s].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # calculate spectral radius
                        cond_num = np.max(np.abs(r.cond_nums[refine]))  # eig_vals[0]:eigenvalues with no grid refinement
                        nelem = r.nelems[refine]
                        nelem_list.append(nelem)
                        cond_num_list.append(cond_num)

                    # plot spectral radius
                    # plt.rcParams.update({'font.size': 22, 'axes.labelsize': 22, 'legend.fontsize': 22,
                    #                      'xtick.labelsize': 22, 'ytick.labelsize': 22})

                    # label_cond = 'SBP- {} $\;$ {} $\;$ $p={}$'.format(pltsetup.sbp_fam[sbp_families[f]], pltsetup.sat_name[sats[s]], p+1)
                    label_cond = '{}'.format(pltsetup.sat_name[sats[s]])
                    plt.plot(nelem_list, cond_num_list, pltsetup.markers[s], linewidth=pltsetup.lw,
                             markersize=pltsetup.ms, label=label_cond)

                    plt.yscale('symlog')
                    plt.xlabel(r'$n_e$')
                    plt.ylabel(r'condition number')
                    plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.legend(ncol=3, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1, loc=4)

                    nelem_list = []
                    cond_num_list = []

                if save_fig:
                    plt.savefig(path + 'cond_nums\\' + 'cond_{}_p{}.pdf'.format(sbp_families[f], p+1),
                                format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()


    # Tabulate number of nonzero elements
    if tabulate_nnz:
        if run_results is None:
            if calc_eigs:
                with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile3:
                    all_results = pickle.load(infile3)
            else:
                with open(path + 'results_poisson2D.pickle', 'rb') as infile3:
                    all_results = pickle.load(infile3)
        else:
            all_results = run_results

        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        nnz_list = []
        row_list = []
        col_list = ['Degree', 'Operator']
        col_list.extend(sats)
        for refine in range(nrefine):
            for p in range(pmin, pmax):
                for f in range(len(sbp_families)):
                    row = [p+1, sbp_families[f]]
                    for s in range(len(sats)):
                        if run_results is None:
                            # SAT and SBP family list in saved data
                            fmod = sbp_famliy_saved_list.index(sbp_families[f])
                            smod = sat_saved_list.index(sats[s])
                        else:
                            fmod = f
                            smod = s

                        # get results from saved tree file
                        res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # add nnz to row of the table
                        row.append(r.nnz_elems[refine])
                    row_list.append(row)
            nnz_list.append(row_list)
            row_list = []

            df = pd.DataFrame(nnz_list[refine], columns=col_list)
            # pd.options.display.float_format = '{:.2e}'.format
            df.to_string(index=False)
            print(df, '\n')
            # print(df.to_latex(), '\n')

    if tabulate_density:
        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        # tabulate relative density of matrix
        nnz_list = []
        row_list = []
        col_list = ['Degree', 'SAT']
        col_list.extend(['gamma', 'est', 'den', 'omega', 'est', 'den', 'diagE', 'est', 'den'])
        for refine in range(nrefine):
            for p in range(pmin, pmax):
                for s in range(len(sats)):
                    row = [p+1, sats[s]]
                    for f in range(len(sbp_families)):
                        if run_results is None:
                            # SAT and SBP family list in saved data
                            fmod = sbp_famliy_saved_list.index(sbp_families[f])
                            smod = sat_saved_list.index(sats[s])
                        else:
                            fmod = f
                            smod = s

                        # get results from saved tree file
                        res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # get nnz of BR1 as a reference
                        resref = all_results.children[0].children[fmod].children[1].children[p].children[0].results
                        rref = SimpleNamespace(**resref)

                        # get theoretical estimate of the nnz
                        nnz_est = nnz_estimate(sbp_families[f], sats[s], p+1, r.nelems[refine])

                        # add relative density to row of the table
                        nnz_elem = r.nnz_elems[refine]
                        row.append(nnz_elem)
                        row.append((nnz_est - nnz_elem)/nnz_elem * 100)
                        density = r.nnz_elems[refine]/rref.nnz_elems[refine]
                        row.append(density)
                    row_list.append(row)
            nnz_list.append(row_list)
            row_list = []

            df = pd.DataFrame(nnz_list[refine], columns=col_list)
            pd.options.display.float_format = '{:.4f}'.format
            df.to_string(index=False)
            print(df, '\n')
            # print(df.to_latex(), '\n')

    if plot_sparsity:
        nrefine = len((all_results.children[0].children[0].children[0].children[0].children[0].results)['hs'])
        nelem_list = []
        nnz_elem_list = []
        nnz_est_list = []
        for p in range(pmin, pmax):
            for s in range(len(sats)):
                for f in range(len(sbp_families)):
                    for refine in range(nrefine):
                        if run_results is None:
                            # SAT and SBP family list in saved data
                            fmod = sbp_famliy_saved_list.index(sbp_families[f])
                            smod = sat_saved_list.index(sats[s])
                        else:
                            fmod = f
                            smod = s

                        # get results from saved tree file
                        res = all_results.children[0].children[fmod].children[smod].children[p].children[0].results
                        r = SimpleNamespace(**res)

                        # get number of nonzeros
                        nnz_elem = r.nnz_elems[refine]
                        nelem = r.nelems[refine]
                        nelem_list.append(nelem)
                        nnz_elem_list.append(nnz_elem)

                        # get theoretical estimate of the nnz
                        nnz_est = nnz_estimate(sbp_families[f], sats[s], p + 1, r.nelems[refine])
                        nnz_est_list.append(nnz_est)

                    # plot
                    plt.rcParams.update({'font.size': 22, 'axes.labelsize': 28, 'legend.fontsize': 22,
                                         'xtick.labelsize': 24, 'ytick.labelsize': 24})

                    label_nnz = 'SBP-{} num.'.format(pltsetup.sbp_fam[sbp_families[f]])
                    label_nnz_est = 'SBP-{} est.'.format(pltsetup.sbp_fam[sbp_families[f]])
                    # label_nnz= '{}'.format(pltsetup.sat_name[sats[s]])
                    plt.plot(nelem_list, r.nnz_elems, pltsetup.markers5[f], linewidth=0, markersize=15, label=label_nnz)
                    plt.plot(nelem_list, nnz_est_list, pltsetup.markers6[f], linewidth=2, markersize=0, label=label_nnz_est)

                    # plt.yscale('symlog')
                    plt.xlabel(r'$n_e$')
                    plt.ylabel(r'$nnz$')
                    # plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.0e}'))
                    # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.0e}'))
                    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    nelem_list = []
                    nnz_elem_list = []
                    nnz_est_list = []

                handles, labels = plt.gca().get_legend_handles_labels()
                order = [0, 2, 4, 1, 3, 5]
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                           ncol=1, labelspacing=0.3, columnspacing=1, handletextpad=0.1)
                if save_fig:
                    plt.savefig(path + 'sparsity\\' + 'nnz_{}_p{}.pdf'.format(sats[s], p+1), format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()
    #######################################

    # res_BR2 = all_results.children[0].children[0].children[0].children[0].children[0].results
    # res_BO = all_results.children[0].children[0].children[1].children[0].children[0].results
    # err_uh = res_BR2['uh'] - res_BO['uh']
    # x_uh = res_BR2['x']
    # y_uh = res_BR2['y']
    # plot_figure_2d(x_uh, y_uh, err_uh)
    # plot_figure_2d(x_uh, y_uh, res_BR2['uh']-res_BR2['u_exact'])
    # plot_figure_2d(x_uh, y_uh, res_BO['uh']-res_BO['u_exact'])
    # plot_figure_2d(x_uh, y_uh, res_BR2['uh'])
    # plot_figure_2d(x_uh, y_uh, res_BO['uh'])
    # plt.show()


    ########################################
    return


def analyze_results_1d(sbp_families=None, sats=None, ps=None, stencil=None, imp=None, prob=None, plot_by_family=False,
                       plot_by_sat=False, plot_spectrum=False, plot_spectral_radius=False, plot_sparsity=False,
                       run_results=None, save_fig=False, plot_adj_by_family=False):

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson1d_results\\'
    if run_results is None:
        # solve and obtain results or open saved from file
        with open(path+'results_poisson1D.pickle', 'rb') as infile:
            all_results = pickle.load(infile)

        pmin = np.min(ps) - 1
        pmax = np.max(ps)
    else:
        all_results = run_results

        pmin = 0
        pmax = len(ps)

    dim = 1
    # setup default values based on input
    sbp_families, sats, ps, degrees, stencil, prob = input_defualt(sbp_families, sats, ps, stencil, prob, dim)

    # setup plot options
    pltsetup_dict = plot_setup(sbp_families, sats, dim, stencil)
    pltsetup = SimpleNamespace(**pltsetup_dict)

    # SAT and SBP family list in saved data
    sat_saved_list = ['BR1', 'BR2', 'LDG', 'BO', 'CNG']
    sbp_famliy_saved_list =  ['CSBP', 'CSBP_Mattsson2004', 'LGL', 'LG', 'HGTL']

    imp_app=[]  # traditional or element type refinement
    for j in range(len(imp)):
        if imp[j] == 'trad':
            imp_app.append(0)
        elif imp[j] == 'elem' and len(imp_app)==1:
            imp_app.append(1)
        elif imp[j] == 'elem' and len(imp_app)==0:
            imp_app.append(0)
    imp_app = sorted(imp_app)

    app=[]  # wide or narrow stencil application
    for j in range(len(stencil)):
        if sten[j] == 'wide':
            app.append(0)
        elif sten[j] == 'narrow' and len(app)==1:
            app.append(1)
        elif sten[j] == 'narrow' and len(app)==0:
            app.append(0)
    app = sorted(app)

    # plot solution by sbp family, i.e., 1 family with varying SAT types
    if plot_by_family:
        for pr in range(len(prob)):
            for i in range(len(imp)):   # traditional or element type refinement
                for p in range(pmin, pmax):
                    for f in range(len(sbp_families)):
                        for s in range(len(sats)):
                            for t in range(len(stencil)):   # wide or narrow stencil
                                if run_results is None:
                                    # SAT and SBP family list in saved data
                                    fmod = sbp_famliy_saved_list.index(sbp_families[f])
                                    smod = sat_saved_list.index(sats[s])
                                else:
                                    fmod = f
                                    smod = s
                                # get results from saved tree file
                                res = all_results.children[0].children[fmod].children[app[t]].children[imp_app[i]].\
                                    children[smod].children[p].children[pr].children[0].results
                                # calculate degrees of freedom and add it to res dictionary
                                res['dof'] = np.asarray([x*y for x, y in zip(res['nelems'], res['ns'])])
                                r = SimpleNamespace(**res)

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = (len(r.dof)) - 3   # -6 gives the 4th step for a 9 step total
                                end = (len(r.dof)) - 0      # -3 gives the 6th step for a 9 step total
                                # calculate solution convergence rates
                                conv_soln = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs[begin:end]), 1)[0])

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = (len(r.dof)) - 3  # -5 gives the 4th step for a 9 step total
                                end = (len(r.dof)) - 0    # -3 gives the 6th step for a 9 step total
                                # calculate functional convergence rates
                                conv_func = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs_func[begin:end]), 1)[0])

                                # plot solution convergence rates
                                plt.figure(1)
                                # plt.loglog(1/r.dof, r.errs, pltsetup.markers[2*s+t], linewidth=pltsetup.lw,
                                #            markersize=pltsetup.ms, label='{}-{}-{} $|$ {} $|$ {} $|$r={:.2f}'.
                                #            format(pltsetup.sbp_fam[sbp_families[f]],
                                #                   pltsetup.stencil_shortname[stencil[t]], imp[i],
                                #                   pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                                fill_type = ['full', 'none', 'full']
                                line_type = [2, 3, 2]
                                marker_type = [pltsetup.markers[2 * s + t], pltsetup.markers1[2 * s + t], pltsetup.markers2[2 * s + t]]
                                markersize_type = [8, pltsetup.ms, pltsetup.ms]
                                plt.loglog(1 / r.dof, r.errs, marker_type[t], linewidth=line_type[t],
                                           markersize=markersize_type[t], fillstyle=fill_type[t], label='{}-{} {} ({:.2f})'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]],
                                                  pltsetup.sat_name[sats[s]], conv_soln))
                                plt.xlabel(r'1/dof')
                                plt.ylabel(r'solution error')
                                plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.4f}'))
                                plt.gca().axes.tick_params(which='minor', width=1, length=6, labelsize=22)
                                plt.gca().axes.tick_params(which='major', width=2, length=12, labelsize=22)
                                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

                                # annotation.slope_marker((0.005, 1e-3),
                                #                         slope=(ps[p]+1)+ 2,
                                #                         size_frac=0.2,
                                #                         text_kwargs={'color': 'k', 'fontsize': 12})
                                if save_fig:
                                    plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarOper_p{}.pdf'.
                                                format(ps[p]), format='pdf', bbox_inches='tight')

                                # plot functional convergence rates
                                plt.figure(2)
                                plt.loglog(1 / r.dof, r.errs_func, marker_type[t], linewidth=line_type[t],
                                           markersize=markersize_type[t], fillstyle=fill_type[t], label='{}-{} {} ({:.2f})'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]],
                                                  pltsetup.sat_name[sats[s]], conv_func))
                                # plt.loglog(1/r.dof, r.errs_func, pltsetup.markers[2*s+t], linewidth=pltsetup.lw,
                                #            markersize=pltsetup.ms, label='{}-{} {}'.
                                #            format(pltsetup.sbp_fam[sbp_families[f]],
                                #             pltsetup.stencil_shortname[stencil[t]],
                                #                   pltsetup.sat_name[sats[s]]))
                                plt.xlabel(r'1/dof')
                                plt.ylabel(r'functional error')
                                plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.4f}'))
                                plt.gca().axes.tick_params(which='minor', width=1, length=6, labelsize=22)
                                plt.gca().axes.tick_params(which='major', width=2, length=12, labelsize=22)
                                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

                                # annotation.slope_marker((0.005, 1e-3),
                                #                         slope=2*(ps[p]),
                                #                         size_frac=0.2,
                                #                         text_kwargs={'color': 'k', 'fontsize': 12})
                                if save_fig:
                                    plt.savefig(path + '\\func_conv_rates\\errs_func_VarOper_p{}.pdf'.
                                                format(ps[p]), format='pdf', bbox_inches='tight')
                plt.show()
                plt.close()

    # plot solution by sat type, i.e., 1 SAT type and varying SBP families
    if plot_by_sat:
        for pr in range(len(prob)):
            for i in range(len(imp)):   # traditional or element type refinement
                for p in range(pmin, pmax):
                    for s in range(len(sats)):
                        for f in range(len(sbp_families)):
                            for t in range(len(stencil)):   # wide or narrow stencil
                                # if not ((sbp_families[f] == 'LGL' or sbp_families[f] == 'LG') and stencil[t]=='narrow'):
                                if run_results is None:
                                    # SAT and SBP family list in saved data
                                    fmod = sbp_famliy_saved_list.index(sbp_families[f])
                                    smod = sat_saved_list.index(sats[s])
                                else:
                                    fmod = f
                                    smod = s
                                    # get results from saved tree file
                                res = all_results.children[0].children[fmod].children[app[t]].children[imp_app[i]]. \
                                    children[smod].children[p].children[pr].children[0].results
                                # calculate degrees of freedom and add it to res dictionary
                                res['dof'] = np.asarray([x * y for x, y in zip(res['nelems'], res['ns'])])
                                r = SimpleNamespace(**res)

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = len(r.dof) - 6
                                end = len(r.dof) - 3
                                # calculate solution convergence rates
                                conv_soln = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs[begin:end]), 1)[0])

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = len(r.dof) - 6
                                end = len(r.dof) - 3
                                # calculate functional convergence rates
                                conv_func = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs_func[begin:end]), 1)[0])

                                # plot solution convergence rates
                                plt.figure(1)
                                # plt.loglog(1/r.dof, r.errs, pltsetup.markers[2*f+t], linewidth=pltsetup.lw,
                                #            markersize=pltsetup.ms, label='{}-{}-{} $|$ {} $|$ {} $|$r={:.2f}'.
                                #            format(pltsetup.sbp_fam[sbp_families[f]],
                                #                   pltsetup.stencil_shortname[stencil[t]], imp[i],
                                #                   pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                                # plt.xlabel(r'$1/dof$')
                                # plt.ylabel(r'solution error')
                                # plt.legend()
                                # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                                # plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)

                                fill_type = ['none', 'full', 'none']
                                line_type = [2, 3, 2, 2, 2]
                                marker_type = [pltsetup.markers[2 * (f) + t], pltsetup.markers1[2 * (f) + t],
                                               pltsetup.markers2[2 * (f) + t]]
                                markersize_type = [16, 8] #, pltsetup.ms, pltsetup.ms]
                                plt.loglog(1 / r.dof, r.errs, marker_type[s], linewidth=line_type[f],
                                           markersize=markersize_type[s], fillstyle=fill_type[s],
                                           label='{}-{} {} ({:.2f})'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]],
                                                  pltsetup.sat_name[sats[s]], conv_soln))
                                plt.xlabel(r'1/dof')
                                plt.ylabel(r'solution error')
                                plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.4f}'))
                                plt.gca().axes.tick_params(which='minor', width=1, length=6, labelsize=22)
                                plt.gca().axes.tick_params(which='major', width=2, length=12, labelsize=22)
                                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

                                if save_fig:
                                    plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarSAT_{}_p{}.pdf'.
                                                format('BO', ps[p]), format='pdf', bbox_inches='tight')

                                # plot functional convergence rates
                                plt.figure(2)
                                # plt.loglog(1/r.dof, r.errs_func, pltsetup.markers[2*f+t], linewidth=pltsetup.lw,
                                #            markersize=pltsetup.ms, label='{}-{}-{} $|${}$|${}$|$ r={:.2f}'.
                                #            format(pltsetup.sbp_fam[sbp_families[f]],
                                #                   pltsetup.stencil_shortname[stencil[t]], imp[i],
                                #                   pltsetup.sat_name[sats[s]], degrees[p], conv_func))
                                # plt.xlabel(r'$1/dof$')
                                # plt.ylabel(r'functional error')
                                # plt.legend()
                                # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                                # plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                                fill_type = ['none', 'full', 'none']
                                line_type = [2, 3, 2, 2, 2]
                                marker_type = [pltsetup.markers[2 * (f) + t], pltsetup.markers1[2 * (f) + t],
                                               pltsetup.markers2[2 * (f) + t]]
                                markersize_type = [16, 8]  # , pltsetup.ms, pltsetup.ms]
                                plt.loglog(1 / r.dof, r.errs_func, marker_type[s], linewidth=line_type[f],
                                           markersize=markersize_type[s], fillstyle=fill_type[s],
                                           label='{}-{} {} ({:.2f})'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]],
                                                  pltsetup.sat_name[sats[s]], conv_func))
                                plt.xlabel(r'1/dof')
                                plt.ylabel(r'functional error')
                                plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.4f}'))
                                plt.gca().axes.tick_params(which='minor', width=1, length=6, labelsize=22)
                                plt.gca().axes.tick_params(which='major', width=2, length=12, labelsize=22)
                                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

                            if save_fig:
                                    plt.savefig(path + '\\func_conv_rates\\errs_func_VarSAT_{}_p{}.pdf'.
                                                format('BO', ps[p]), format='pdf', bbox_inches='tight')
                    plt.show()
                    plt.close()

    #######################################

    # res_BR2= all_results.children[0].children[0].children[0].children[0]. \
    #     children[0].children[0].children[0].children[0].results
    # res_BO = all_results.children[0].children[0].children[0].children[0]. \
    #     children[1].children[0].children[0].children[0].results
    # err_uh = res_BR2['uh']-res_BO['uh']
    # x_uh = res_BR2['x'].reshape((-1, 1), order='F')
    # plt.plot(x_uh, err_uh)
    # plt.show()

    ########################################


    # plot of condition number, sparsity, and eigenvalue spectrum can be done in a similar fashion as the 2D case
    # plot solution by sbp family, i.e., 1 family with varying SAT types
    if plot_adj_by_family:
        for pr in range(len(prob)):
            for i in range(len(imp)):  # traditional or element type refinement
                for p in range(pmin, pmax):
                    for f in range(len(sbp_families)):
                        for s in range(len(sats)):
                            for t in range(len(stencil)):  # wide or narrow stencil
                                if run_results is None:
                                    # SAT and SBP family list in saved data
                                    fmod = sbp_famliy_saved_list.index(sbp_families[f])
                                    smod = sat_saved_list.index(sats[s])
                                else:
                                    fmod = f
                                    smod = s
                                # get results from saved tree file
                                res = all_results.children[0].children[fmod].children[app[t]].children[imp_app[i]]. \
                                    children[smod].children[p].children[pr].children[0].results
                                # calculate degrees of freedom and add it to res dictionary
                                res['dof'] = np.asarray([x * y for x, y in zip(res['nelems'], res['ns'])])
                                r = SimpleNamespace(**res)

                                # set refinement levels where the convergence rates calculation begins and ends
                                begin = (len(r.dof)) - 4  # -6 gives the 4th step for a 9 step total
                                end = (len(r.dof)) - 1  # -3 gives the 6th step for a 9 step total
                                # calculate solution convergence rates
                                conv_soln = np.abs(np.polyfit(np.log10(r.dof[begin:end]),
                                                              np.log10(r.errs_adj[begin:end]), 1)[0])

                                # plot solution convergence rates
                                plt.figure(1)
                                # plt.loglog(1/r.dof, r.errs, pltsetup.markers[2*s+t], linewidth=pltsetup.lw,
                                #            markersize=pltsetup.ms, label='{}-{}-{} $|$ {} $|$ {} $|$r={:.2f}'.
                                #            format(pltsetup.sbp_fam[sbp_families[f]],
                                #                   pltsetup.stencil_shortname[stencil[t]], imp[i],
                                #                   pltsetup.sat_name[sats[s]], degrees[p], conv_soln))
                                fill_type = ['full', 'none', 'full']
                                line_type = [2, 3, 2]
                                marker_type = [pltsetup.markers[2 * s + t], pltsetup.markers1[2 * s + t],
                                               pltsetup.markers2[2 * s + t]]
                                markersize_type = [8, pltsetup.ms, pltsetup.ms]
                                plt.loglog(1 / r.dof, r.errs_adj, marker_type[f], linewidth=line_type[f],
                                           markersize=markersize_type[f], fillstyle=fill_type[f],
                                           label='{}-{} {} ({:.2f})'.
                                           format(pltsetup.sbp_fam[sbp_families[f]],
                                                  pltsetup.stencil_shortname[stencil[t]],
                                                  pltsetup.sat_name[sats[s]], conv_soln))
                                plt.xlabel(r'1/dof')
                                plt.ylabel(r'adjoint error')
                                plt.legend(ncol=2, labelspacing=0.1, columnspacing=0.7, handletextpad=0.1)
                                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
                                # plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.4f}'))
                                plt.gca().axes.tick_params(which='minor', width=1, length=6, labelsize=22)
                                plt.gca().axes.tick_params(which='major', width=2, length=12, labelsize=22)
                                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9]))

                                # annotation.slope_marker((0.005, 1e-3),
                                #                         slope=(ps[p]+1)+ 2,
                                #                         size_frac=0.2,
                                #                         text_kwargs={'color': 'k', 'fontsize': 12})
                                if save_fig:
                                    plt.savefig(path + '\\adj_conv_rates\\adj_soln_VarOper_p{}.pdf'.
                                                format(ps[p]), format='pdf', bbox_inches='tight')

                plt.show()
                plt.close()

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
    markers = marker_shapes = marker_lines = marker_all = []

    if dim == 2:
        # 2D families
        if 'gamma' in sbp_families:
            sbp_fam['gamma'] = '$\Gamma$'
        if 'omega' in sbp_families:
            sbp_fam['omega'] = '$\Omega$'
        if 'diage' in sbp_families:
            sbp_fam['diage'] = 'E'

        markers = ['--Db', ':Xk', '-.or', '-.<g', ':xm', ':dc', '--s']
        markers1 = ['--*b', '--xk', '--or', '--<g', '--Xm', '--dc', '--s']
        markers2 = ['-*b', '-xk', '-or', '-<g', '-Xm', '-dc', '-s']
        markers3 = [':*b', ':xk', ':or', ':<g', ':Xm', ':dc', ':s']
        markers4 = ['-.*b', '-.xk', '-.or', '-.<g', '-.Xm', '-.dc', '-.s']
        markers5 = ['-*b', '-xk', '-or', '-<g', '-Xm', '-dc', '-s']
        markers6 = [':*b', '-.Xk', '--or', ':<g', ':Xm', ':dc', ':s']
        markers_all = [markers1, markers2, markers3, markers4]
        marker_lines = ['--', '-', ':', '-.']
        marker_shapes = ['dg', 'Xk', 'or', '<b', 'xm', '*c', 's']

    elif dim == 1:
        # 1D families
        if 'CSBP' in sbp_families:
            sbp_fam['CSBP'] = 'CSBP1'
        if 'CSBP_Mattsson2004' in sbp_families:
            sbp_fam['CSBP_Mattsson2004'] = 'CSBP'
        if 'CSBP_Mattsson2013' in sbp_families:
            sbp_fam['CSBP_Mattsson2013'] = 'CSBP2'
        if 'HGTL' in sbp_families:
            sbp_fam['HGTL'] = 'HGTL'
        if 'HGT' in sbp_families:
            sbp_fam['HGT'] = 'HGT'
        if 'LG' in sbp_families:
            sbp_fam['LG'] = 'LG'
        if 'LGL' in sbp_families:
            sbp_fam['LGL'] = 'LGL'
        if 'LGL-Dense' in sbp_families:
            sbp_fam['LGL-Dense'] = 'LGL-Dense'

        if 'wide' in stencil:
            stencil_shortname['wide'] = 'W'
        if 'narrow' in stencil:
            stencil_shortname['narrow'] = 'N'

        markers = [':sg', ':sg', ':*k', ':*k', ':or', ':or', ':<b', ':b', '--dm', '-dm', ':hc', '-hc', '--X','-X']
        # markers = ['--sg', '-sg', '--*y', '-*y', ':or', '-or', '-.<b', '-<b', '--dm', '-dm', ':hc', '-hc', '--X','-X']
        markers1 = ['-.sg', '-.sg', '-.*k', '-.*k', '-.or', '-.or', '-.<b', '-.<b', '--dm', '-dm', ':hc', '-hc', '--X','-X']
        markers2 = ['-xg', '-xg', '-+k', '-+k', '-2r', '-2r', '-1b', '-1b', '--dm', '-dm', ':hc', '-hc', '--X', '-X']
        # markers1 = ['--*b', '--xk', '--or', '--<g', '--Xm', '--dc', '--s']
        # markers2 = ['-*b', '-xk', '-or', '-<g', '-Xm', '-dc', '-s']
        markers3 = ['-.<b', '-.<b', '-.hm', '-.hm', '-.db', '-.db', '-.<c', '-.<c', '--dm', '-dm', ':hc', '-hc', '--X','-X']
        markers4 = ['-.*b', '-.xk', '-.or', '-.<g', '-.Xm', '-.dc', '-.s']
        markers5 = ['-*b', '-xk', '-or', '-<g', '-Xm', '-dc', '-s']
        markers6 = [':*b', '-.Xk', '--or', ':<g', ':Xm', ':dc', ':s']
        markers_all = [markers1, markers2, markers3, markers4]
        marker_lines = ['--', '-', ':', '-.']
        marker_shapes = ['dg', 'Xk', 'or', '<b', 'xm', '*c', 's']

    # dictionary to hold names of SATs
    sat_name = {}
    if 'BO' in sats:
        sat_name['BO']  = " BO $\;\,$ "
    if 'BR2' in sats:
        sat_name['BR2'] = " BR2$\,$ "
    if 'BR1' in sats:
        sat_name['BR1'] = " BR1$\,$ "
    if 'LDG' in sats:
        sat_name['LDG'] = " LDG "
    if 'CDG' in sats:
        sat_name['CDG'] = " CDG "
    if 'CNG' in sats:
        sat_name['CNG'] = " CNG "
    if 'IP' in sats:
        sat_name['IP']  = "SIPG "
    if 'NIPG' in sats:
        sat_name['NIPG'] ="NIPG "

    # set plot parameters
    params = {'axes.labelsize': 28,
              'legend.fontsize': 24,
              'xtick.labelsize': 24,
              'ytick.labelsize': 24,
              'text.usetex': False,         # True works only if results are read from pickle saved file
              'font.family': 'serif',
              'figure.figsize': [12,8]} #[12,6],[6,8]
    plt.rcParams.update(params)
    lw = 4  # lineweight
    ms = 15  # markersize

    return {'sbp_fam': sbp_fam, 'sat_name': sat_name, 'markers': markers, 'params': params, 'lw': lw, 'ms': ms,
            'stencil_shortname': stencil_shortname, 'markers_all': markers_all, 'marker_lines': marker_lines,
            'marker_shapes': marker_shapes, 'markers1': markers1, 'markers2': markers2, 'markers3': markers3,
            'markers4': markers4, 'markers5': markers5, 'markers6': markers6}


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

def nnz_estimate(sbp_family, sat, p, nelems, dim=2):
    nf = p+1
    d = dim
    sbp_family = sbp_family.lower()
    nnodes = Ref2D_SBP.nodes_sbp_2d(p, sbp_family)['nnodes']
    nnz_est = 0
    if sbp_family == 'omega':
        if sat == 'BR1':
            nnz_est = (d**2 + 2*d + 2)*nnodes**2*nelems
        elif sat == 'BR2' or sat == 'SIPG' or sat == 'BO' or sat == 'NIPG' or sat == 'CDG' or sat == 'CNG':
            nnz_est = (d+2)*nnodes**2*nelems
        elif sat == 'LDG':
            # nnz_est = (d**2 + 2)*(nnodes**2)*nelems
            nnz_est = (np.ceil(nelems/(d+1))*(d**2 + 2) + np.floor(d*nelems/(d+1))*(d**2 + 1))*nnodes**2
    elif sbp_family == 'gamma':
        if sat == 'BR1':
            nnz_est = (nnodes**2 + (d+1)*(2*nnodes*nf - nf**2) + (d**2 + d)*nf**2) * nelems
        elif sat == 'BR2' or sat == 'SIPG' or sat == 'BO' or sat == 'NIPG':
            nnz_est = (nnodes**2 + (d+1)*(2*nnodes*nf - nf**2))*nelems
        elif sat == 'CDG' or sat == 'CNG':
            nnz_est = (nnodes**2 + (d+1)*nnodes*nf)*nelems
        elif sat == 'LDG':
            # nnz_est = (nnodes**2 + (d+1)*nnodes*nf + (d**2 - d)*nf**2)*nelems
            nnz_est = (nnodes**2*nelems+ (d+1)*nnodes*nf*nelems + ((d**2 + 1)*nelems - np.ceil(nelems/(d+1))*(d+1)
                                                      - np.floor(d*nelems/(d+1))*(d+2))*nf**2)
    elif sbp_family == 'diage':
        if sat == 'BR1' or sat == 'BR2' or sat == 'SIPG' or sat == 'BO' or sat == 'NIPG':
            nnz_est = (nnodes**2 + (d+1)*(2*nnodes*nf - nf**2))*nelems
        elif sat == 'LDG' or sat == 'CDG' or sat == 'CNG':
            nnz_est = (nnodes**2 + (d+1)*nnodes*nf)*nelems

    return nnz_est

# ================================================  2D-plots  ======================================================== #
# give parameters for 2D solver and analyzer
# fam = ['gamma', 'omega', 'diage']
# sat = ['BR1', 'BR2', 'LDG', 'CDG', 'BO', 'CNG']
# p = [1,2,3,4]
fam = ['omega']
sat = ['BR2', 'CNG']
p = [4]
p_map = 2

# ------ plots --------
plt_fam = True
plt_sat = False
plt_sat_all = False
plt_adj_fam = True
plt_adj_sat = False
plt_eig = False
plt_rho = False
plt_cond = False
plt_sparsity = False
showMesh = False
plt_soln = False
# ------- tables -----
tab_cond = False
tab_density = False
tab_nnz = False
# ------ save --------
save_figure = False
save_runs = False
modify_saved = False

# ------ solve --------
adj = True
calc_eigs = False
calc_cond_num = False
curve_mesh = True

# soln = None
# soln = save_results(h=3, nrefine=3, sats=sat, sbp_families=fam, ps=p, solve_adjoint=adj, save_results=save_runs,
#                     calc_cond=calc_cond_num, calc_eigvals=calc_eigs, showMesh=showMesh, p_map=p_map, curve_mesh=curve_mesh,
#                     plot_fig=plt_soln, modify_saved=modify_saved)
# analyze_results_2d(sats=sat, sbp_families=fam, ps=p, plot_by_family=plt_fam, plot_by_sat=plt_sat,  plot_by_sat_all=plt_sat_all,
#                    plot_spectrum=plt_eig, plot_spectral_radius=plt_rho, plot_sparsity=plt_sparsity,
#                    plot_adj_by_sat=plt_adj_sat, plot_adj_by_family=plt_adj_fam, tabulate_cond_num=tab_cond,
#                    tabulate_density = tab_density, tabulate_nnz = tab_nnz, run_results=soln, save_fig=save_figure)
# ==================================================================================================================== #

# ===============================================   1D-plots  ======================================================== #
# give parameters for 1D solver and analyzer
opers = ['CSBP_Mattsson2013']
# opers = ['CSBP', 'CSBP_Mattsson2004', 'CSBP_Mattsson2013', 'LGL', 'LG', 'HGTL']
# sat = ['BR2', 'LDG', 'BO', 'CNG']
# sat = ['BR2','LDG', 'BO', 'CNG']
sat = ['BR2', 'CNG']
p = [4]
# p = [1, 2, 3, 4]
sten = ['narrow']
# sten = ['wide', 'narrow']
degree = ['p1', 'p2 ', 'p3', 'p4']
# app = ['wide', 'narrow']
# app =['wide']
# imp_type = ['trad', 'elem']
imp_type = ['elem']
prob_type = ['Diff']

adj = True
plt_adj_fam = True
plt_fam = True
plt_sat = False
plt_eig = False
plt_rho = False
plt_sparsity = False
calc_eigs = False
save_figure = False

soln = None
soln = save_results(nrefine=7, sbp_families=opers, sats=sat, ps=p, solve_adjoint=adj, save_results=True,
                 calc_cond=False, calc_eigvals=False, dim=1, stencil= sten, imp=imp_type, prob=prob_type)
analyze_results_1d(sats=sat, sbp_families=opers, ps=p, stencil=sten, imp=imp_type, prob=prob_type, plot_by_family=plt_fam,
                   plot_by_sat=plt_sat, plot_spectrum=plt_eig, plot_spectral_radius=plt_rho, plot_sparsity=plt_sparsity,
                   run_results=soln, save_fig=save_figure, plot_adj_by_family=plt_adj_fam)
# ==================================================================================================================== #
