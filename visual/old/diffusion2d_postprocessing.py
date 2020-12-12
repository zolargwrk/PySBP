import numpy as np
from types import SimpleNamespace
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from solver.diffusion_solver import poisson_sbp_2d
from matplotlib.ticker import MaxNLocator

class Node(object):
    """Data structure to contain the SBP results. The node structure is laid out below:
                                                                    results
                                                              -------------------
    Node level 1                                                  sbp_family
                                                             -------------------
                                                            /          |         \
    Node level 2                                      gamma         omega        diagE ....
                                                      /               |              \
                                            _____________       _____________        ______________
                                          |     |   |   |       |    |   |   |        |    |   |    |
    Node level 3                        BR1   BR2  LDG  BO...  BR1  BR2  LDG BO...   BR1  BR2  LDG  BO ...
                                        /      |   |   |       |    |    |   |       |    |    |    |
                                 __________   ___
                               /  /   /   /    |
    Node level 4             p1  p2  p3  p4   degree....
                             /   /   /  /     |
                          data   ...
                     in dictionary

    """
    def __init__(self, value):
        """Initialize a node and the levels under it.
            e.g., for the root node 'sbp_family' we set data0='gamma', data1='omega', data2='diage', then under
            data0='gamma', we will have another level for the SATs as data0='BR1', etc."""
        self.value = value
        self.data0 = None
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None
        self.data6 = None
        self.data7 = None
        self.data8 = None
        self.data9 = None
        self.children = []

    def add_child(self, data):
        """This is used to add the final results of dictionary to the corresponding parent node of degrees"""
        self.children.append(data)

    def add_node(self, data):
        """Add node to the current node level"""
        if self.data0 is None:
            self.data0 = Node(data)
        elif self.data1 is None:
            self.data1 = Node(data)
        elif self.data2 is None:
            self.data2 = Node(data)
        elif self.data3 is None:
            self.data3 = Node(data)
        elif self.data4 is None:
            self.data4 = Node(data)
        elif self.data5 is None:
            self.data5 = Node(data)
        elif self.data6 is None:
            self.data6 = Node(data)
        elif self.data7 is None:
            self.data7 = Node(data)
        elif self.data8 is None:
            self.data8 = Node(data)
        elif self.data9 is None:
            self.data9 = Node(data)

    def find_node(self, value):
        """Find node by value in the current node level"""
        if self.data0 is not None and self.data0.value == value:
            return self.data0
        elif self.data1 is not None and self.data1.value == value:
            return self.data1
        elif self.data2 is not None and self.data2.value == value:
            return self.data2
        elif self.data3 is not None and self.data3.value == value:
            return self.data3
        elif self.data4 is not None and self.data4.value == value:
            return self.data4
        elif self.data5 is not None and self.data5.value == value:
            return self.data5
        elif self.data6 is not None and self.data6.value == value:
            return self.data6
        elif self.data7 is not None and self.data7.value == value:
            return self.data7
        elif self.data8 is not None and self.data8.value == value:
            return self.data8
        elif self.data9 is not None and self.data9.value == value:
            return self.data9


class DataTree(object):
    """Initializes a data tree"""
    def __init__(self, value):
        self.root = Node(value)


def save_results(h=0.8, nrefine=2, sbp_families=None, sats=None, ps=None, solve_adjoint=True, save_results=True,
                 calc_cond=False, calc_eigvals=False):

    if sbp_families is None:
        sbp_families = ['gamma', 'omega', 'diagE']
    if sats is None:
        sats = ['BR2']
    if ps is None:
        ps = [1, 2, 3, 4]
        degree = ['p1', 'p2', 'p3', 'p4']
    else:
        degree = []
        for d in range(0, len(ps)):
            degree.append('p' + str(ps[d]))

    sbp_families = [x.lower() for x in sbp_families]
    sats = [x.upper() for x in sats]

    # create a data tree
    results = DataTree('sbp_family')

    for family in range(0, len(sbp_families)):
        # add node to the tree to specify the sbp family
        results.root.add_node(sbp_families[family])

        for sat in range(0, len(sats)):
            # find and add node to the tree to specify the SAT type
            found_node1 = results.root.find_node(sbp_families[family])
            found_node1.add_node(sats[sat])

            for p in range(0, len(ps)):
                # add node to specify the degree of the operator
                found_node2 = found_node1.find_node(sats[sat])
                found_node2.add_node(degree[p])

                # solve the Poisson problem and obtain data
                soln = poisson_sbp_2d(ps[p], h, nrefine, sbp_families[family], sats[sat], solve_adjoint, plot_fig=False,
                                      calc_condition_num=calc_cond, calc_eigvals=calc_eigvals)

                # add data to the leaves of the tree
                found_node3 = found_node2.find_node(degree[p])
                found_node3.add_child(soln)

    # save result
    if save_results:
        path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\'
        with open(path+'results_poisson2D_eigvals.pickle', 'wb') as outfile:
            pickle.dump(results, outfile)
        
    return results


def analyze_results(sbp_families=None, sats=None, ps=None, plot_by_family=False, plot_by_sat=False, plot_spectrum=False,
                    plot_spectral_radius=False, plot_sparsity=False, run_results=None, save_fig=False):

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\'
    if run_results is None:
        # solve and obtain results or open saved from file
        with open(path+'results_poisson2D_v2.pickle', 'rb') as infile:
            results = pickle.load(infile)
    else:
        results = run_results

    if sbp_families is None:
        sbp_families = ['gamma', 'omega', 'diagE']
        # sbp_fam = ['$\Gamma$', '$\Omega$', 'E']
    if sats is None:
        sats = ['BR2']
    if ps is None:
        ps = [1, 2, 3, 4]
        degree = ['p1', 'p2', 'p3', 'p4']
    else:
        degree = []
        for d in range(0, len(ps)):
            degree.append('p' + str(ps[d]))

    sbp_families = [x.lower() for x in sbp_families]
    sats = [x.upper() for x in sats]

    sbp_fam ={}
    if 'gamma' in sbp_families:
        sbp_fam['gamma'] = '$\Gamma$'
    if 'omega' in sbp_families:
        sbp_fam['omega'] = '$\Omega$'
    if 'diage' in sbp_families:
        sbp_fam['diage'] = 'E'

    sat_name={}
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
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [12, 9]}
    plt.rcParams.update(params)
    lw = 3      # lineweight
    ms = 15     # markersize
    markers = ['--*g', '--sy', ':<r', '-.ob', '--dm', ':hc', '--X']

    # plot solution by sbp family
    if plot_by_family:
        for p in range(0, len(ps)):
            for family in range(0, len(sbp_families)):
                plt.figure(1)
                plt.figure(2)
                # find node that specify the sbp family
                f = results.root.find_node(sbp_families[family])

                for sat in range(0, len(sats)):
                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    # find node that specify the degree of the operator
                    d = s.find_node(degree[p])

                    # get results saved for each degree
                    res = d.children[0]
                    r = SimpleNamespace(**res)
                    begin = len(r.hs) - 3
                    end = len(r.hs)
                    # begin = -2
                    # end = -1

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])
                    # conv_soln = (np.log10(r.errs_soln[end]) - np.log10(r.errs_soln[begin])) \
                    #            / (np.log10(r.hs[end]) - np.log10(r.hs[begin]))

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])
                    # conv_func = (np.log10(r.errs_func[end]) - np.log10(r.errs_func[begin])) \
                    #            / (np.log10(r.hs[end]) - np.log10(r.hs[begin]))

                    # plot solution convergence rates
                    plt.figure(1)
                    plt.loglog(r.hs, r.errs_soln, markers[sat], linewidth=lw, markersize=ms,
                               label='SBP-{}| {}| {}| r={:.2f}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], degree[p], conv_soln))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in solution')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    # plt.gca().axes.set_aspect(aspect=0.4)

                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarOper_{}_p{}.pdf'.format(sbp_families[family], ps[p]), format='pdf')


                    # plot functional convergence rates
                    plt.figure(2)
                    plt.loglog(r.hs, r.errs_func, markers[sat], linewidth=lw, markersize=ms,
                               label='SBP-{}| {}| {}| r={:.2f}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], degree[p], conv_func))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in functional')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + '\\func_conv_rates\\errs_func_VarOper_{}_p{}.pdf'.format(sbp_families[family], ps[p]), format='pdf')

                plt.show()


    # plot adjoint by spb family
    if plot_by_sat:
        for p in range(0, len(ps)):
            for family in range(0, len(sbp_families)):
                for sat in range(0, len(sats)):
                    # find node that specify the sbp family
                    f = results.root.find_node(sbp_families[family])

                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    # find node that specify the degree of the operator
                    d = s.find_node(degree[p])

                    # get results saved for each degree
                    res = d.children[0]
                    r = SimpleNamespace(**res)
                    begin = len(r.hs) - 3
                    end = len(r.hs)
                    # begin = -2
                    # end = -1

                    # plot adjoint convergence rates
                    if r.errs_adj:
                        conv_adj = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_adj[begin:end]), 1)[0])
                        # conv_adj = (np.log10(r.errs_adj[end]) - np.log10(r.errs_adj[begin])) \
                        #            / (np.log10(r.hs[end]) - np.log10(r.hs[begin]))
                        plt.figure()
                        plt.loglog(r.hs, r.errs_adj, markers[sat], linewidth=lw, markersize=ms,
                                   label='SBP-{}| {}| {}| r={:.2f}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], degree[p], conv_adj))
                        plt.xlabel(r'$h$')
                        plt.ylabel(r'error in adjoint')
                        plt.legend()
                        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                        plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                        plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                        if save_fig:
                            plt.savefig(path + 'soln_conv_rates\\errs_adj_VarOper_{}_p{}.pdf'.format(sats[sat], ps[p]), format='pdf')
                        plt.show()
                        plt.close()

    # plot solution by sat type
    if plot_by_sat:
        for p in range(0, len(ps)):
            plt.figure(3)
            plt.figure(4)
            for sat in range(0, len(sats)):
                for family in range(0, len(sbp_families)):
                    # find node that specify the sbp family
                    f = results.root.find_node(sbp_families[family])
                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    # find node that specify the degree of the operator
                    d = s.find_node(degree[p])

                    # get results saved for each degree
                    res = d.children[0]
                    r = SimpleNamespace(**res)
                    begin = len(r.hs) - 3
                    end = len(r.hs)
                    # begin = -2
                    # end = -1

                    # calculate solution convergence rates
                    conv_soln = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_soln[begin:end]), 1)[0])
                    # conv_soln = (np.log10(r.errs_soln[end]) - np.log10(r.errs_soln[begin]))\
                    #             /(np.log10(r.hs[end]) - np.log10(r.hs[begin]))

                    # calculate functional convergence rates
                    conv_func = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_func[begin:end]), 1)[0])
                    # conv_func = (np.log10(r.errs_func[end]) - np.log10(r.errs_func[begin])) \
                    #             / (np.log10(r.hs[end]) - np.log10(r.hs[begin]))

                    # plot solution convergence rates
                    plt.figure(3)
                    plt.loglog(r.hs, r.errs_soln, markers[family], linewidth=lw, markersize=ms,
                               label='SBP-{}| {}| {}| r={:.2f}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], degree[p], conv_soln))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in solution')
                    plt.legend()
                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarSAT_{}_p{}.pdf'.format(sats[sat], ps[p]), format='pdf')

                    # plot functional convergence rates
                    plt.figure(4)
                    plt.loglog(r.hs, r.errs_func, markers[family], linewidth=lw, markersize=ms,
                               label='SBP-{}| {}| {}| r={:.2f}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], degree[p], conv_func))
                    plt.xlabel(r'$h$')
                    plt.ylabel(r'error in functional')
                    plt.legend()

                    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                    plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                    if save_fig:
                        plt.savefig(path + 'func_conv_rates\\errs_func_VarSAT_{}_p{}.pdf'.format(sats[sat], ps[p]), format='pdf')

                plt.show()

    # plot adjoint by sat type
    if plot_by_sat:
        for p in range(0, len(ps)):
            for sat in range(0, len(sats)):
                for family in range(0, len(sbp_families)):
                    # find node that specify the sbp family
                    f = results.root.find_node(sbp_families[family])

                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    # find node that specify the degree of the operator
                    d = s.find_node(degree[p])

                    # get results saved for each degree
                    res = d.children[0]
                    r = SimpleNamespace(**res)
                    begin = len(r.hs) - 3
                    end = len(r.hs)
                    # begin = -2
                    # end = -1

                    # plot adjoint convergence rates
                    if r.errs_adj:
                        conv_adj = np.abs(np.polyfit(np.log10(r.hs[begin:end]), np.log10(r.errs_adj[begin:end]), 1)[0])
                        # conv_adj = (np.log10(r.errs_adj[end]) - np.log10(r.errs_adj[begin])) \
                        #             / (np.log10(r.hs[end]) - np.log10(r.hs[begin]))
                        plt.figure()
                        plt.loglog(r.hs, r.errs_adj, markers[family], linewidth=lw, markersize=ms,
                                   label='SBP-{}, {}, {}, r={:.2f}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], degree[p], conv_adj))
                        plt.xlabel(r'$h$')
                        plt.ylabel(r'error in adjoint')
                        plt.legend()
                        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                        plt.gca().xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
                        plt.gca().axes.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
                        if save_fig:
                            plt.savefig(path + 'soln_conv_rates\\errs_adj_VarOper_{}_p{}.pdf'.format(sbp_families[family], ps[p]), format='pdf')
                        plt.show()
                        plt.close()

    # plot spectrum of the system matrix
    if plot_spectrum:
        if run_results is None:
            with open(path + 'results_poisson2D_eigvals.pickle', 'rb') as infile2:
                results_eig_vals = pickle.load(infile2)
        else:
            results_eig_vals = run_results

        for p in range(0, len(ps)):
            for family in range(0, len(sbp_families)):
                plt.figure(1)
                # find node that specify the sbp family
                f = results_eig_vals.root.find_node(sbp_families[family])
                for sat in range(0, len(sats)):
                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    # find node that specify the degree of the operator
                    d = s.find_node(degree[p])

                    # get results saved for each degree
                    res = d.children[0]
                    r = SimpleNamespace(**res)

                    # get real and imaginary parts
                    X = [x.real for x in r.eig_vals]
                    Y = [x.imag for x in r.eig_vals]

                    # plot eigenvalue spectrum
                    plt.rcParams.update({'font.size': 22, 'axes.labelsize': 22, 'legend.fontsize': 22,
                                         'xtick.labelsize': 22, 'ytick.labelsize': 22})
                    plt.figure(1)
                    plt.scatter(X, Y, s=120,
                               label='SBP- {} $|$ {} $|$  p{} $|$  $\lambda_L$= ${:.2f}$ $|$  $\lambda_S$ = ${:.2e}$'.
                                format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], ps[p], np.max(X), np.min(X)))
                    plt.xlabel(r'$Re(\lambda)$')
                    plt.ylabel(r'$Im(\lambda)$')
                    plt.legend()
                    # plt.gca().axes.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    plt.xscale('symlog')

                if save_fig:
                    plt.savefig(path + 'spectrum\\' + 'spectrum_{}_p{}.pdf'.format(sbp_families[family], ps[p]), format='pdf')
                # plt.show()
                plt.close()

    if plot_spectral_radius:
        if run_results is None:
            with open(path + 'results_poisson2D.pickle', 'rb') as infile3:
                results_eig_vals = pickle.load(infile3)
        else:
            results_eig_vals = run_results

        rholist = []
        plist = []
        for family in range(0, len(sbp_families)):
            plt.figure(1)
            # find node that specify the sbp family
            f = results.root.find_node(sbp_families[family])

            for refine in range(0, 4):
                for sat in range(0, len(sats)):
                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    for p in range(0, len(ps)):
                        # find node that specify the degree of the operator
                        d = s.find_node(degree[p])

                        # get results saved for each degree
                        res = d.children[0]
                        r = SimpleNamespace(**res)

                        # calculate spectral radius
                        rho = np.max(np.abs(r.eig_vals[refine])) # eig_vals[0] is the eigenvalues calculated before any grid refinement
                        nelem = r.nelems[refine]
                        rholist.append(rho)
                        plist.append(p+1)

                    # plot spectral radius
                    plt.rcParams.update({'font.size': 28, 'axes.labelsize': 28, 'legend.fontsize': 28,
                                         'xtick.labelsize': 28, 'ytick.labelsize': 28})
                    plt.semilogy(plist, rholist, markers[sat], linewidth=lw*3/4, markersize=ms*4/3,
                                label='SBP- {} $|$ {} $|$ $n_e$={}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], nelem))
                    plt.xlabel(r'operator degree, $p$')
                    plt.ylabel(r'spectral radius, $\max(|{\lambda}|)$')
                    plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.legend()

                    plist=[]
                    rholist=[]

                if save_fig:
                    plt.savefig(path + 'spectral_radius\\' + 'spectral_radius_{}_{}.pdf'.format(sbp_families[family], nelem),
                                format='pdf')
                plt.show()
                # plt.close()

    if plot_sparsity:
        if run_results is None:
            with open(path + 'results_poisson2D.pickle', 'rb') as infile4:
                results_eig_vals = pickle.load(infile4)
        else:
            results_eig_vals = run_results

        nnz_per_nelems = []
        plist =[]
        for family in range(0, len(sbp_families)):
            plt.figure(1)
            # find node that specify the sbp family
            f = results.root.find_node(sbp_families[family])

            for refine in range(0, 4):
                for sat in range(0, len(sats)):
                    # find node that specify the SAT type
                    s = f.find_node(sats[sat])

                    for p in range(0, len(ps)):
                        # find node that specify the degree of the operator
                        d = s.find_node(degree[p])

                        # get results saved for each degree
                        res = d.children[0]
                        r = SimpleNamespace(**res)

                        # calculate spectral radius
                        nnz_elems = np.max(np.abs(r.nnz_elems[refine])) # eig_vals[0] is the eigenvalues calculated before any grid refinement
                        nelem = r.nelems[refine]
                        nnz_per_nelems.append(nnz_elems/nelem)
                        plist.append(p+1)

                    # plot spectral radius
                    plt.rcParams.update({'font.size': 28, 'axes.labelsize': 28, 'legend.fontsize': 28,
                                         'xtick.labelsize': 28, 'ytick.labelsize': 28})
                    plt.plot(plist, nnz_per_nelems, markers[sat], linewidth=lw*3/4, markersize=ms*4/3,
                                label='SBP- {} $|$ {} $|$ $n_e$={}'.format(sbp_fam[sbp_families[family]], sat_name[sats[sat]], nelem))
                    plt.xlabel(r'operator degree, $p$')
                    plt.ylabel(r'$nnz(A)/n_{elems})$')
                    plt.gca().axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.legend()

                    plist=[]
                    nnz_per_nelems=[]

                if save_fig:
                    plt.savefig(path + 'sparsity\\' + 'sparsity_{}_{}.pdf'.format(sbp_families[family], nelem),
                                format='pdf')
                plt.show()
                # plt.close()
    return


fam = ['gamma', 'omega', 'diagE']
sat = ['BR1', 'BR2', 'LDG', 'CDG', 'BO', 'CNG']
p = [1, 2, 3, 4]
# fam = ['diage']
# sat = ['BR1', 'BR2']
# sat = ['LDG', 'CDG']
# p = [4]
adj = False
plt_fam = False
plt_sat = False
plt_eig = False
plt_rho = False
plt_sparsity = True
calc_eigs = False
save_figure = True

soln = None
# soln = save_results(h=0.5, nrefine=1, sats=sat, sbp_families=fam, ps=p, solve_adjoint=adj, save_results=True,
#                     calc_cond=False, calc_eigvals=calc_eigs)
analyze_results(sats=sat, sbp_families=fam, ps=p, plot_by_family=plt_fam, plot_by_sat=plt_sat, plot_spectrum=plt_eig,
                plot_spectral_radius=plt_rho, plot_sparsity=plt_sparsity, run_results=None, save_fig=save_figure)
