from types import SimpleNamespace
from solver.problem_statements import advec_diff1D_problem_input
from solver.advection_diffusion_solver import advec_diff_1d
import json


def dict_to_sns(d):
    return SimpleNamespace(**d)


def save_results():
    opers = ['CSBP', 'CSBP_Mattsson2004', 'HGTL']
    SATs = ['BR1', 'BR2', 'LDG', 'BO', 'CNG']
    refines = ['trad', 'ntrad']
    ps = [1, 2, 3, 4]
    apps = [1, 2]
    stencil = ['wide', 'narrow']
    degree = ['p1', 'p2', 'p3', 'p4']

    results_dict_trad = {}      # results with traditional refinement
    results_dict_ntrad = {}     # results with element type (non-traditional) refinement
    oper_dict = {}
    sat_dict = {}
    p_dict = {}
    app_dict = {}

    for oper in range(0, 3):
        for sat in range(0, 5):
            for p in range(0, 4):
                for app in range(0, 2):
                    sol = advec_diff_1d(ps[p], 0, 1, 2, opers[oper], 'upwind', SATs[sat], 2, 'ntrad',
                                        'nPeriodic', 'sbp_sat', advec_diff1D_problem_input, n=25, app=apps[app])

                    sols = SimpleNamespace(**sol)
                    if sols.b != 0:
                        if sols.a != 0:
                            prob = 'AdvDiff'    # Advection-Diffusion
                        elif sols.a == 0:
                            prob = 'Diff'       # Diffusion
                    elif sols.b == 0:
                        if sols.a != 0:
                            prob = 'Adv'        # Advection

                    app_dict["{0}".format(stencil[app])] = sol.copy()
                    p_dict["{0}".format(degree[p])] = app_dict.copy()
                    sat_dict["{0}".format(SATs[sat])] = p_dict.copy()
                    oper_dict["{0}".format(opers[oper])] = sat_dict.copy()
                    results_dict_ntrad["{0}".format(prob)] = oper_dict.copy()

    with open('results_ntrad.txt', 'w') as outfile:
        json.dump(results_dict_ntrad, outfile)

    for oper in range(0, 3):
        for p in range(0, 4):
            for app in range(0, 2):
                sol = advec_diff_1d(ps[p], 0, 1, 1, opers[oper], 'upwind', 'BR1', 2, 'trad',
                                    'nPeriodic', 'sbp_sat', advec_diff1D_problem_input, n=25, app=apps[app])

                sols = SimpleNamespace(**sol)
                if sols.b != 0:
                    if sols.a != 0:
                        prob = 'AdvDiff'  # Advection-Diffusion
                    elif sols.a == 0:
                        prob = 'Diff'  # Diffusion
                elif sols.b == 0:
                    if sols.a != 0:
                        prob = 'Adv'  # Advection

                app_dict["{0}".format(stencil[app])] = sol.copy()
                p_dict["{0}".format(degree[p])] = app_dict.copy()
                oper_dict["{0}".format(opers[oper])] = p_dict.copy()
                results_dict_trad["{0}".format(prob)] = oper_dict.copy()

    with open('results_trad.txt', 'w') as outfile:
        json.dump(results_dict_trad, outfile)

    with open('results_trad.txt') as fd:
        res_trad = json.load(fd, object_hook=dict_to_sns)

    return

sols = save_results()