import numpy as np
from types import SimpleNamespace
import pickle
import matplotlib.pyplot as plt
from solver.diffusion_solver import poisson_sbp_2d


class Node(object):
    """Data structure to contain the SBP results. The node structure is layed out below:
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
    Node level 4             p1  p2  p3  p4   ps....
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


def save_results(h=0.8, nrefine=2, sbp_families=None, sats=None, ps=None):

    if sbp_families is None:
        sbp_families = ['gamma', 'omega', 'diagE']
        # sbp_fam = ['$\Gamma$', '$\Omega$', 'E']
    if sats is None:
        sats = ['BR2']
    if ps is None:
        ps = ['p1', 'p2', 'p3', 'p4']

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

            for p in range(1, len(ps)+1):
                # add node to specify the degree of the operator
                found_node2 = found_node1.find_node(sats[sat])
                found_node2.add_node(ps[p-1])

                # solve the Poisson problem and obtain data
                soln = poisson_sbp_2d(p, h, nrefine, sbp_family=sbp_families[family], flux_type=sats[sat])

                # add data to the leaves of the tree
                found_node3 = found_node2.find_node(ps[p-1])
                found_node3.add_child(soln)

    # save result
    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\'
    with open(path+'results_poisson2D.pickle', 'wb') as outfile:
        pickle.dump(results, outfile)
        
    return results


def analyze_results(sbp_families=None, sats=None, ps=None):
    # solve and obtain results or open saved from file
    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\'
    with open(path+'results_poisson2D.pickle', 'rb') as infile:
        results = pickle.load(infile)

    if sbp_families is None:
        sbp_families = ['gamma', 'omega', 'diagE']
        # sbp_fam = ['$\Gamma$', '$\Omega$', 'E']
    if sats is None:
        sats = ['BR2']
    if ps is None:
        ps = ['p1', 'p2', 'p3', 'p4']

    sbp_families = [x.lower() for x in sbp_families]
    sats = [x.upper() for x in sats]

    sbp_fam =[]
    if 'gamma' in sbp_families:
        sbp_fam.append('$\Gamma$')
    if 'omega' in sbp_families:
        sbp_fam.append('$\Omega$')
    if 'diage' in sbp_families:
        sbp_fam.append('E')

    # set plot parameters
    params = {'axes.labelsize': 25,
              'legend.fontsize': 25,
              'xtick.labelsize': 25,
              'ytick.labelsize': 25,
              'text.usetex': False,
              'font.family': 'serif',
              'figure.figsize': [12, 9]}
    plt.rcParams.update(params)
    lw = 3      # lineweight
    ms = 12     # markersize
    markers = ['--*g', '--sy', '--<r', '--ob', '--Dm']

    # plot solution error
    for p in range(1, len(ps)+1):
        for family in range(0, len(sbp_families)):
            # find node that specify the sbp family
            f = results.root.find_node(sbp_families[family])

            for sat in range(0, len(sats)):
                # find node that specify the SAT type
                s = f.find_node(sats[sat])

                # find node that specify the degree of the operator
                d = s.find_node(ps[p-1])

                # get results saved for each degree
                res = d.children[0]
                r = SimpleNamespace(**res)

                # calculate convergence rates
                conv = np.abs(np.polyfit(np.log10(r.hs[1:-1]), np.log10(r.errs_soln[1:-1]), 1)[0])

                # plot result
                plt.loglog(r.hs, r.errs_soln, markers[family], linewidth=lw, markersize=ms,
                           label='SBP-{}, {}, {}, r={:.2f}'.format(sbp_fam[family], sats[sat], ps[p-1], conv))
                plt.xlabel(r'$h$')
                plt.ylabel(r'error in solution')
                plt.legend()
                plt.savefig(path + '\\soln_conv_rates\\errs_soln_VarOper_{}_p{}.pdf'.format(sats[sat], p), format='pdf')
        plt.show()
        # plt.close()

soln = save_results(h=0.5, nrefine=5, sats=['BR1'], sbp_families=['gamma'])
analyze_results(sats=['BR1'], sbp_families=['gamma'])

