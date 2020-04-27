import numpy as np
from types import SimpleNamespace
from solver.problem_statements import advec_diff1D_problem_input
from solver.advection_diffusion_solver import advec_diff_1d
import json
import csv
import matplotlib.pyplot as plt
from mpltools import annotation


def dict_to_sns(d):
    return SimpleNamespace(**d)


def save_results():
    opers = ['CSBP', 'CSBP_Mattsson2004', 'HGTL']
    SATs = ['BR1', 'BR2', 'LDG', 'BO', 'CNG']
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

    for oper in range(0, len(opers)):
        for sat in range(0, len(SATs)):
            for p in range(0, len(ps)):
                for app in range(0, len(apps)):
                    sol = advec_diff_1d(ps[p], 0, 1, 2, opers[oper], 'upwind', SATs[sat], 8, 'ntrad',
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

    with open('results_ntrad_a0b1.txt', 'w') as outfile:
        json.dump(results_dict_ntrad, outfile)

    for oper in range(0, 3):
        for p in range(0, 4):
            for app in range(0, 2):
                sol = advec_diff_1d(ps[p], 0, 1, 1, opers[oper], 'upwind', 'BR1', 8, 'trad',
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

    with open('results_trad_a0b1.txt', 'w') as outfile:
        json.dump(results_dict_trad, outfile)

    # with open('results_ntrad.txt') as fd:
    #     res_ntrad = json.load(fd, object_hook=dict_to_sns)

    return

sols = save_results()

def plot_cond(resN_list, resT_list):
    # resN_list = [probN, operN, satN, psN, stencilN, conv_solN, conv_adjN, conv_funcN, nsN, nelemsN, errsN, errs_funcN, cond_numN, cond_numT_all, norm_condN]
    # resT_list = [probT, operT, psT, stencilT, conv_solT, conv_adjT, conv_funcT, nsT, errsT, errs_funcT, cond_numT]

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\pysbp_results\\advec_diff_results\\figures\\cond_figs\\'
    norm_condN = resN_list[-1]
    cond_numN = resN_list[-3]
    norm_condT = resT_list[-1]
    dof = np.asarray(resN_list[8]) * np.asarray(resN_list[9])

    for k in range(len(resT_list[1])):
        if resT_list[1][k] == 'CSBP':
            resT_list[1][k] = 'CSBP1'
        elif resT_list[1][k] == 'CSBP_Mattsson2004':
            resT_list[1][k] = 'CSBP2'

    for k in range(len(resN_list[1])):
        if resN_list[1][k] == 'CSBP':
            resN_list[1][k] = 'CSBP1'
        elif resN_list[1][k] == 'CSBP_Mattsson2004':
            resN_list[1][k] = 'CSBP2'

    params = {'axes.labelsize': 20,
              'legend.fontsize': 16,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': False,
              'font.family': 'serif',
              'figure.figsize':[12, 9]}
    plt.rcParams.update(params)

    lw = 2
    ms = 10
    for k0 in range(0, 240, 120):       # we consider 2 problems, i.e.,  data is divided in two where total data is 240
        for k1 in range(0, 80, 40):     # 2 operators CSBP1 and CSBP2 hold the first 0-80 and 120-200 data entries
            for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
                plt.loglog(dof[k0 + k1 + 0 + p], norm_condN[k0 + k1 + 0 + p], '--*g', linewidth=lw,   label='{}, {}'.format( resN_list[2][k0 + k1 +  0  + p], resN_list[4][k0 + k1 +  0  + p]), markersize=ms+3)
                plt.loglog(dof[k0 + k1 + 1 + p], norm_condN[k0 + k1 + 1 + p], '-*g',  linewidth=lw,  label='{}, {}'.format( resN_list[2][k0 + k1 +  1  + p], resN_list[4][k0 + k1 +  1  + p]), markersize=ms+3)
                plt.loglog(dof[k0 + k1 + 8 + p], norm_condN[k0 + k1 + 8 + p], '--sy',  linewidth=lw,  label='{}, {}'.format( resN_list[2][k0 + k1 +  8  + p], resN_list[4][k0 + k1 +  8  + p]), markersize=ms)
                plt.loglog(dof[k0 + k1 + 9 + p], norm_condN[k0 + k1 + 9 + p],'-sy',  linewidth=lw,   label='{}, {}'.format( resN_list[2][k0 + k1 +  9  + p], resN_list[4][k0 + k1 +  9  + p]), markersize=ms)
                plt.loglog(dof[k0 + k1 + 16 + p], norm_condN[k0 + k1 + 16 + p],'--<r',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  16 + p], resN_list[4][k0 + k1 +  16 + p]), markersize=ms)
                plt.loglog(dof[k0 + k1 + 17 + p], norm_condN[k0 + k1 + 17 + p],'-<r',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  17 + p], resN_list[4][k0 + k1 +  17 + p]), markersize=ms)
                plt.loglog(dof[k0 + k1 + 24 + p], norm_condN[k0 + k1 + 24 + p],'--ob',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  24 + p], resN_list[4][k0 + k1 +  24 + p]), markersize=ms+2)
                plt.loglog(dof[k0 + k1 + 25 + p], norm_condN[k0 + k1 + 25 + p],'-ob',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  25 + p], resN_list[4][k0 + k1 +  25 + p]), markersize=ms+2)
                plt.loglog(dof[k0 + k1 + 32 + p], norm_condN[k0 + k1 + 32 + p],'--Dm',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  32 + p], resN_list[4][k0 + k1 +  32 + p]), markersize=ms)
                plt.loglog(dof[k0 + k1 + 33 + p], norm_condN[k0 + k1 + 33 + p],'-Dm',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  33 + p], resN_list[4][k0 + k1 +  33 + p]), markersize=ms)

                plt.xlabel(r'$n\times n_{elem}$')
                plt.ylabel(r'ratio of condition numbers, ${A_{elem}}/{A_{trad}}$')
                plt.legend()
                plt.savefig(path + 'Nn_cond_{}_{}_p{}.pdf'.format(resN_list[0][k0 + k1 + 0 + p], resN_list[1][k0 + k1 + 0 + p], resN_list[3][k0 + k1 +  0  + p]), format='pdf')
                plt.close()
                # plt.show()

    for k0 in range(0, 240, 120):       # we consider 2 problems, i.e.,  data is divided in two where total data is 240
        for k1 in range(0, 80, 40):     # 2 operators CSBP1 and CSBP2 hold the first 0-80 and 120-200 data entries
            for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
                plt.loglog(resN_list[10][k0 + k1 + 0 + p], norm_condN[k0 + k1 + 0 + p], '--*g', linewidth=lw,   label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  0  + p], resN_list[2][k0 + k1 +  0  + p], resN_list[4][k0 + k1 +  0  + p]), markersize=ms+3)
                plt.loglog(resN_list[10][k0 + k1 + 1 + p], norm_condN[k0 + k1 + 1 + p], '-*g',  linewidth=lw,  label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  1  + p], resN_list[2][k0 + k1 +  1  + p], resN_list[4][k0 + k1 +  1  + p]), markersize=ms+3)
                plt.loglog(resN_list[10][k0 + k1 + 8 + p], norm_condN[k0 + k1 + 8 + p], '--sy',  linewidth=lw,  label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  8  + p], resN_list[2][k0 + k1 +  8  + p], resN_list[4][k0 + k1 +  8  + p]), markersize=ms)
                plt.loglog(resN_list[10][k0 + k1 + 9 + p], norm_condN[k0 + k1 + 9 + p],'-sy',  linewidth=lw,   label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  9  + p], resN_list[2][k0 + k1 +  9  + p], resN_list[4][k0 + k1 +  9  + p]), markersize=ms)
                plt.loglog(resN_list[10][k0 + k1 + 16 + p], norm_condN[k0 + k1 + 16 + p],'--<r',  linewidth=lw, label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  16 + p], resN_list[2][k0 + k1 +  16 + p], resN_list[4][k0 + k1 +  16 + p]), markersize=ms)
                plt.loglog(resN_list[10][k0 + k1 + 17 + p], norm_condN[k0 + k1 + 17 + p],'-<r',  linewidth=lw, label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  17 + p], resN_list[2][k0 + k1 +  17 + p], resN_list[4][k0 + k1 +  17 + p]), markersize=ms)
                plt.loglog(resN_list[10][k0 + k1 + 24 + p], norm_condN[k0 + k1 + 24 + p],'--ob',  linewidth=lw, label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  24 + p], resN_list[2][k0 + k1 +  24 + p], resN_list[4][k0 + k1 +  24 + p]), markersize=ms+2)
                plt.loglog(resN_list[10][k0 + k1 + 25 + p], norm_condN[k0 + k1 + 25 + p],'-ob',  linewidth=lw, label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  25 + p], resN_list[2][k0 + k1 +  25 + p], resN_list[4][k0 + k1 +  25 + p]), markersize=ms+2)
                plt.loglog(resN_list[10][k0 + k1 + 32 + p], norm_condN[k0 + k1 + 32 + p],'--Dm',  linewidth=lw, label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  32 + p], resN_list[2][k0 + k1 +  32 + p], resN_list[4][k0 + k1 +  32 + p]), markersize=ms)
                plt.loglog(resN_list[10][k0 + k1 + 33 + p], norm_condN[k0 + k1 + 33 + p],'-Dm',  linewidth=lw, label='$p={}$, {}, {}'.format(resN_list[3][k0 + k1 +  33 + p], resN_list[2][k0 + k1 +  33 + p], resN_list[4][k0 + k1 +  33 + p]), markersize=ms)

                plt.xlabel('error in solution')
                plt.ylabel('condition number')
                plt.legend()
                plt.savefig(path + 'Nerrsol_cond_{}_{}_p{}.pdf'.format(resN_list[0][k0 + k1 + 0 + p], resN_list[1][k0 + k1 + 0 + p], resN_list[3][k0 + k1 +  0  + p]), format='pdf')
                plt.close()
                # plt.show()


    for k0 in range(0, 240, 120):       # we consider 2 problems, i.e.,  data is divided in two where total data is 240
        for k1 in range(0, 80, 40):     # 2 operators CSBP1 and CSBP2 hold the first 0-80 and 120-200 data entries
            for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
                plt.loglog(resN_list[11][k0 + k1 + 0 + p], cond_numN[k0 + k1 + 0 + p], '--*g', linewidth=lw,   label='{}, {}'.format( resN_list[2][k0 + k1 +  0  + p], resN_list[4][k0 + k1 +  0  + p]), markersize=ms+3)
                plt.loglog(resN_list[11][k0 + k1 + 1 + p], cond_numN[k0 + k1 + 1 + p], '-*g',  linewidth=lw,  label='{}, {}'.format( resN_list[2][k0 + k1 +  1  + p], resN_list[4][k0 + k1 +  1  + p]), markersize=ms+3)
                plt.loglog(resN_list[11][k0 + k1 + 8 + p], cond_numN[k0 + k1 + 8 + p], '--sy',  linewidth=lw,  label='{}, {}'.format( resN_list[2][k0 + k1 +  8  + p], resN_list[4][k0 + k1 +  8  + p]), markersize=ms)
                plt.loglog(resN_list[11][k0 + k1 + 9 + p], cond_numN[k0 + k1 + 9 + p],'-sy',  linewidth=lw,   label='{}, {}'.format( resN_list[2][k0 + k1 +  9  + p], resN_list[4][k0 + k1 +  9  + p]), markersize=ms)
                plt.loglog(resN_list[11][k0 + k1 + 16 + p], cond_numN[k0 + k1 + 16 + p],'--<r',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  16 + p], resN_list[4][k0 + k1 +  16 + p]), markersize=ms)
                plt.loglog(resN_list[11][k0 + k1 + 17 + p], cond_numN[k0 + k1 + 17 + p],'-<r',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  17 + p], resN_list[4][k0 + k1 +  17 + p]), markersize=ms)
                plt.loglog(resN_list[11][k0 + k1 + 24 + p], cond_numN[k0 + k1 + 24 + p],'--ob',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  24 + p], resN_list[4][k0 + k1 +  24 + p]), markersize=ms+2)
                plt.loglog(resN_list[11][k0 + k1 + 25 + p], cond_numN[k0 + k1 + 25 + p],'-ob',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  25 + p], resN_list[4][k0 + k1 +  25 + p]), markersize=ms+2)
                plt.loglog(resN_list[11][k0 + k1 + 32 + p], cond_numN[k0 + k1 + 32 + p],'--Dm',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  32 + p], resN_list[4][k0 + k1 +  32 + p]), markersize=ms)
                plt.loglog(resN_list[11][k0 + k1 + 33 + p], cond_numN[k0 + k1 + 33 + p],'-Dm',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  33 + p], resN_list[4][k0 + k1 +  33 + p]), markersize=ms)

                plt.xlabel('error in output functional')
                plt.ylabel('condition number')
                plt.legend()
                plt.savefig(path + 'Nerrfunc_cond_{}_{}_p{}.pdf'.format(resN_list[0][k0 + k1 + 0 + p], resN_list[1][k0 + k1 + 0 + p], resN_list[3][k0 + k1 +  0  + p]), format='pdf')
                plt.close()
                # plt.show()

    for k1 in range(0, 48, 24):     # we consider 2 problems, i.e.,  data is divided in two where total data is 48, and two operators, CSBP1 and CSBP2, occupying data from 0-16 and 24-40
        for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), norm_condT[k1 + 0 + p], '--*k', linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  0  + p], resT_list[3][k1 +  0  + p]), markersize=ms+1)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), norm_condT[k1 + 1 + p], '-*k',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  1  + p], resT_list[3][k1 +  1  + p]), markersize=ms+1)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), norm_condT[k1 + 8 + p], '--sc',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  8  + p], resT_list[3][k1 +  8  + p]), markersize=ms-3)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), norm_condT[k1 + 9 + p],'-sc',  linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  9  + p], resT_list[3][k1 +  9  + p]), markersize=ms-3)

            plt.xlabel(r'$n$')
            plt.ylabel('condition number')
            plt.legend()
            plt.savefig(path + 'Tn_cond_{}_{}_p{}.pdf'.format(resT_list[0][k1 + 0 + p], resT_list[1][k1 + 0 + p], resT_list[2][k1 +  0  + p]), format='pdf')
            plt.close()
            # plt.show()


    for k1 in range(0, 48, 24):     # we consider 2 problems, i.e.,  data is divided in two where total data is 48, and two operators, CSBP1 and CSBP2, occupying data from 0-16 and 24-40
        for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
            plt.loglog(resT_list[8][k1 + 0 + p], norm_condT[k1 + 0 + p], '--*k', linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  0  + p], resT_list[3][k1 +  0  + p]), markersize=ms+1)
            plt.loglog(resT_list[8][k1 + 1 + p], norm_condT[k1 + 1 + p], '-*k',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  1  + p], resT_list[3][k1 +  1  + p]), markersize=ms+1)
            plt.loglog(resT_list[8][k1 + 8 + p], norm_condT[k1 + 8 + p], '--sc',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  8  + p], resT_list[3][k1 +  8  + p]), markersize=ms-3)
            plt.loglog(resT_list[8][k1 + 9 + p], norm_condT[k1 + 9 + p],'-sc',  linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  9  + p], resT_list[3][k1 +  9  + p]), markersize=ms-3)

            plt.xlabel('error in solution')
            plt.ylabel('condition number')
            plt.legend()
            plt.savefig(path + 'Terrsol_cond_{}_{}_p{}.pdf'.format(resT_list[0][k1 + 0 + p], resT_list[1][k1 + 0 + p], resT_list[2][k1 +  0  + p]), format='pdf')
            plt.close()
            # plt.show()

    for k1 in range(0, 48, 24):     # we consider 2 problems, i.e.,  data is divided in two where total data is 48, and two operators, CSBP1 and CSBP2, occupying data from 0-16 and 24-40
        for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
            plt.loglog(resT_list[9][k1 + 0 + p], norm_condT[k1 + 0 + p], '--*k', linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  0  + p], resT_list[3][k1 +  0  + p]), markersize=ms+1)
            plt.loglog(resT_list[9][k1 + 1 + p], norm_condT[k1 + 1 + p], '-*k',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  1  + p], resT_list[3][k1 +  1  + p]), markersize=ms+1)
            plt.loglog(resT_list[9][k1 + 8 + p], norm_condT[k1 + 8 + p], '--sc',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  8  + p], resT_list[3][k1 +  8  + p]), markersize=ms-3)
            plt.loglog(resT_list[9][k1 + 9 + p], norm_condT[k1 + 9 + p],'-sc',  linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  9  + p], resT_list[3][k1 +  9  + p]), markersize=ms-3)

            plt.xlabel('error in output functional')
            plt.ylabel('condition number')
            plt.legend()
            plt.savefig(path + 'Terrfunc_cond_{}_{}_p{}.pdf'.format(resT_list[0][k1 + 0 + p], resT_list[1][k1 + 0 + p], resT_list[2][k1 +  0  + p]), format='pdf')
            plt.close()
            # plt.show()

    return

def plot_conv(resN_list, resT_list):
    # resN_list = [probN, operN, satN, psN, stencilN, conv_solN, conv_adjN, conv_funcN, nsN, nelemsN, errsN, errs_funcN, cond_numN, cond_numT_all, norm_condN]
    # resT_list = [probT, operT, psT, stencilT, conv_solT, conv_adjT, conv_funcT, nsT, errsT, errs_funcT, cond_numT]

    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\pysbp_results\\advec_diff_results\\figures\\err_figs\\'
    dof = np.asarray(resN_list[8]) * np.asarray(resN_list[9])

    for k in range(len(resT_list[1])):
        if resT_list[1][k] == 'CSBP':
            resT_list[1][k] = 'CSBP1'
        elif resT_list[1][k] == 'CSBP_Mattsson2004':
            resT_list[1][k] = 'CSBP2'

    for k in range(len(resN_list[1])):
        if resN_list[1][k] == 'CSBP':
            resN_list[1][k] = 'CSBP1'
        elif resN_list[1][k] == 'CSBP_Mattsson2004':
            resN_list[1][k] = 'CSBP2'

    params = {'axes.labelsize': 16,
              'legend.fontsize': 12,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'text.usetex': False,
              'font.family': 'serif',
              'figure.figsize':[8, 6]}
    plt.rcParams.update(params)

    lw = 1.5
    ms = 6
    color1 = (0.4, 0.8, 1)

    for k0 in range(0, 240, 120):       # we consider 2 problems, i.e.,  data is divided in two where total data is 240
        for k1 in range(0, 80, 40):     # 2 operators CSBP1 and CSBP2 hold the first 0-80 and 120-200 data entries
            for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
                plt.loglog(1/dof[k0 + k1 + 0 + p], resN_list[10][k0 + k1 + 0 + p], '--*g', linewidth=lw,   label='{}, {}'.format( resN_list[2][k0 + k1 +  0  + p], resN_list[4][k0 + k1 +  0  + p]), markersize=ms+3)
                plt.loglog(1/dof[k0 + k1 + 1 + p], resN_list[10][k0 + k1 + 1 + p], '-*g',  linewidth=lw,  label='{}, {}'.format( resN_list[2][k0 + k1 +  1  + p], resN_list[4][k0 + k1 +  1  + p]), markersize=ms+3)
                plt.loglog(1/dof[k0 + k1 + 8 + p], resN_list[10][k0 + k1 + 8 + p], '--sy',  linewidth=lw,  label='{}, {}'.format( resN_list[2][k0 + k1 +  8  + p], resN_list[4][k0 + k1 +  8  + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 9 + p], resN_list[10][k0 + k1 + 9 + p],'-sy',  linewidth=lw,   label='{}, {}'.format( resN_list[2][k0 + k1 +  9  + p], resN_list[4][k0 + k1 +  9  + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 16 + p], resN_list[10][k0 + k1 + 16 + p],'--<r',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  16 + p], resN_list[4][k0 + k1 +  16 + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 17 + p], resN_list[10][k0 + k1 + 17 + p],'-<r',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  17 + p], resN_list[4][k0 + k1 +  17 + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 24 + p], resN_list[10][k0 + k1 + 24 + p],'--ob',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  24 + p], resN_list[4][k0 + k1 +  24 + p]), markersize=ms+2)
                plt.loglog(1/dof[k0 + k1 + 25 + p], resN_list[10][k0 + k1 + 25 + p],'-ob',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  25 + p], resN_list[4][k0 + k1 +  25 + p]), markersize=ms+2)
                plt.loglog(1/dof[k0 + k1 + 32 + p], resN_list[10][k0 + k1 + 32 + p],'--Dm',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  32 + p], resN_list[4][k0 + k1 +  32 + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 33 + p], resN_list[10][k0 + k1 + 33 + p],'-Dm',  linewidth=lw, label='{}, {}'.format( resN_list[2][k0 + k1 +  33 + p], resN_list[4][k0 + k1 +  33 + p]), markersize=ms)

                annotation.slope_marker((1/dof[k0 + k1 + 17 + p][5], resN_list[10][k0 + k1 + 17 + p][5]-(3/4)*resN_list[10][k0 + k1 + 17 + p][5]),
                                        slope=resN_list[3][k0 + k1 + 0 + p] + 2,
                                        size_frac=0.12,
                                        text_kwargs={'color': 'k', 'fontsize':12},
                                        poly_kwargs={'facecolor':  color1})

                annotation.slope_marker((1/dof[k0 + k1 + 32 + p][4], 1.5/4*resN_list[10][k0 + k1 + 32 + p][3]),
                                        slope=resN_list[3][k0 + k1 + 0 + p] + 1,
                                        size_frac=0.12,
                                        invert=True,
                                        text_kwargs={'color': 'k', 'fontsize':12},
                                        poly_kwargs={'facecolor':  color1})

                plt.xlabel(r'$1/n_{elem}$')
                plt.ylabel('error in solution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + 'Nerrsol_{}_{}_p{}.pdf'.format(resN_list[0][k0 + k1 + 0 + p], resN_list[1][k0 + k1 + 0 + p], resN_list[3][k0 + k1 +  0  + p]), format='pdf')
                plt.close()
                # plt.show()

    for k0 in range(0, 240, 120):       # we consider 2 problems, i.e.,  data is divided in two where total data is 240
        for k1 in range(0, 80, 40):     # 2 operators CSBP1 and CSBP2 hold the first 0-80 and 120-200 data entries
            for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
                plt.loglog(1/dof[k0 + k1 + 0 + p], resN_list[11][k0 + k1 + 0 + p], '--*g', linewidth=lw,   label='{}, {}'.format(resN_list[2][k0 + k1 +  0  + p], resN_list[4][k0 + k1 +  0  + p]), markersize=ms+3)
                plt.loglog(1/dof[k0 + k1 + 1 + p], resN_list[11][k0 + k1 + 1 + p], '-*g',  linewidth=lw,  label='{}, {}'.format(resN_list[2][k0 + k1 +  1  + p], resN_list[4][k0 + k1 +  1  + p]), markersize=ms+3)
                plt.loglog(1/dof[k0 + k1 + 8 + p], resN_list[11][k0 + k1 + 8 + p], '--sy',  linewidth=lw,  label='{}, {}'.format(resN_list[2][k0 + k1 +  8  + p], resN_list[4][k0 + k1 +  8  + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 9 + p], resN_list[11][k0 + k1 + 9 + p],'-sy',  linewidth=lw,   label='{}, {}'.format(resN_list[2][k0 + k1 +  9  + p], resN_list[4][k0 + k1 +  9  + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 16 + p], resN_list[11][k0 + k1 + 16 + p],'--<r',  linewidth=lw, label='{}, {}'.format(resN_list[2][k0 + k1 +  16 + p], resN_list[4][k0 + k1 +  16 + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 17 + p], resN_list[11][k0 + k1 + 17 + p],'-<r',  linewidth=lw, label='{}, {}'.format(resN_list[2][k0 + k1 +  17 + p], resN_list[4][k0 + k1 +  17 + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 24 + p], resN_list[11][k0 + k1 + 24 + p],'--ob',  linewidth=lw, label='{}, {}'.format(resN_list[2][k0 + k1 +  24 + p], resN_list[4][k0 + k1 +  24 + p]), markersize=ms+2)
                plt.loglog(1/dof[k0 + k1 + 25 + p], resN_list[11][k0 + k1 + 25 + p],'-ob',  linewidth=lw, label='{}, {}'.format(resN_list[2][k0 + k1 +  25 + p], resN_list[4][k0 + k1 +  25 + p]), markersize=ms+2)
                plt.loglog(1/dof[k0 + k1 + 32 + p], resN_list[11][k0 + k1 + 32 + p],'--Dm',  linewidth=lw, label='{}, {}'.format(resN_list[2][k0 + k1 +  32 + p], resN_list[4][k0 + k1 +  32 + p]), markersize=ms)
                plt.loglog(1/dof[k0 + k1 + 33 + p], resN_list[11][k0 + k1 + 33 + p],'-Dm',  linewidth=lw, label='{}, {}'.format(resN_list[2][k0 + k1 +  33 + p], resN_list[4][k0 + k1 +  33 + p]), markersize=ms)

                if resN_list[3][k0 + k1 + 0 + p] == 1:
                    c = 0.45
                elif resN_list[3][k0 + k1 + 0 + p]==2:
                    c = 0.65
                elif resN_list[3][k0 + k1 + 0 + p]==3:
                    c = 1.2
                else:
                    c = 2.85

                annotation.slope_marker((c/dof[k0 + k1 + 17 + p][5], resN_list[10][k0 + k1 + 17 + p][6]-(3.5/4)*resN_list[10][k0 + k1 + 17 + p][6]),
                                            slope= 2*resN_list[3][k0 + k1 + 0 + p],
                                            size_frac=0.12,
                                            text_kwargs={'color': 'k', 'fontsize':12},
                                            poly_kwargs={'facecolor':  color1})

                plt.xlabel(r'$1/n_{elem}$')
                plt.ylabel('error in output functional')
                plt.legend()
                plt.savefig(path + 'Nerrfunc_{}_{}_p{}.pdf'.format(resN_list[0][k0 + k1 + 0 + p], resN_list[1][k0 + k1 + 0 + p], resN_list[3][k0 + k1 +  0  + p]), format='pdf')
                plt.close()
                # plt.show()

    for k1 in range(0, 48, 24):     # we consider 2 problems, i.e.,  data is divided in two where total data is 48, and two operators, CSBP1 and CSBP2, occupying data from 0-16 and 24-40
        for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[8][k1 + 0 + p], '--*k', linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  0  + p], resT_list[3][k1 +  0  + p]), markersize=ms+1)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[8][k1 + 1 + p], '-*k',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  1  + p], resT_list[3][k1 +  1  + p]), markersize=ms+1)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[8][k1 + 8 + p], '--sc',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  8  + p], resT_list[3][k1 +  8  + p]), markersize=ms-3)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[8][k1 + 9 + p],'-sc',  linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  9  + p], resT_list[3][k1 +  9  + p]), markersize=ms-3)

            annotation.slope_marker((1/np.asarray(resT_list[7][k1 + 1 + p][6]), np.asarray(resT_list[8][k1 + 1 + p][6])-(2/4)*np.asarray(resT_list[8][k1 + 1 + p][6])),
                                        slope=np.asarray(resT_list[2][k1 + 0 + p])+ 2,
                                        size_frac=0.12,
                                        text_kwargs={'color': 'k', 'fontsize':12},
                                        poly_kwargs={'facecolor':  color1})

            annotation.slope_marker((1/np.asarray(resT_list[7][k1 + 0 + p][5]), np.asarray(resT_list[8][k1 + 0 + p][5])+(3/4)*np.asarray(resT_list[8][k1 + 0 + p][5])),
                                        slope=np.asarray(resT_list[2][k1 + 0 + p])+ 1,
                                        size_frac=0.12,
                                        invert=True,
                                        text_kwargs={'color': 'k', 'fontsize':12},
                                        poly_kwargs={'facecolor': color1})

            plt.xlabel(r'$1/n$')
            plt.ylabel('error in solution')
            plt.legend()
            plt.savefig(path + 'Terrsol_{}_{}_p{}.pdf'.format(resT_list[0][k1 + 0 + p], resT_list[1][k1 + 0 + p], resT_list[2][k1 +  0  + p]), format='pdf')
            plt.close()
            # plt.show()

    for k1 in range(0, 48, 24):     # we consider 2 problems, i.e.,  data is divided in two where total data is 48, and two operators, CSBP1 and CSBP2, occupying data from 0-16 and 24-40
        for p in range(0, 8, 2):    # first 8 places taken by BR1 used with p1, p2, p3, p4, then BR2, etc.
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[9][k1 + 0 + p], '--*k', linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  0  + p], resT_list[3][k1 +  0  + p]), markersize=ms+1)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[9][k1 + 1 + p], '-*k',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  1  + p], resT_list[3][k1 +  1  + p]), markersize=ms+1)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[9][k1 + 8 + p], '--sc',  linewidth=lw,  label='{}, {}'.format(resT_list[1][k1 +  8  + p], resT_list[3][k1 +  8  + p]), markersize=ms-3)
            plt.loglog(1/np.asarray(resT_list[7][k1 + 0 + p]), resT_list[9][k1 + 9 + p],'-sc',  linewidth=lw,   label='{}, {}'.format(resT_list[1][k1 +  9  + p], resT_list[3][k1 +  9  + p]), markersize=ms-3)

            annotation.slope_marker((1.2/np.asarray(resT_list[7][k1 + 9 + p][6]), np.asarray(resT_list[9][k1 + 9 + p][6])-(2/4)*np.asarray(resT_list[9][k1 + 9 + p][6])),
                                            slope= 2*resT_list[2][k1 + 0 + p],
                                            size_frac=0.12,
                                            text_kwargs={'color': 'k', 'fontsize':12},
                                            poly_kwargs={'facecolor':  color1})
            plt.xlabel(r'$1/n$')
            plt.ylabel('error in output functional')
            plt.legend()
            plt.savefig(path + 'Terrfunc_{}_{}_p{}.pdf'.format(resT_list[0][k1 + 0 + p], resT_list[1][k1 + 0 + p], resT_list[2][k1 +  0  + p]), format='pdf')
            plt.close()
            # plt.show()

    return


def analyze_results():
    # load data
    path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\advec_diff_results\\'

    resN_a1b1 = json.load(open(path + 'results_ntrad_a1b1.txt'), object_hook=dict_to_sns)
    resT_a1b1 = json.load(open(path + 'results_trad_a1b1.txt'), object_hook=dict_to_sns)
    resN_a1b1e4 = json.load(open(path + 'results_ntrad_a1b1e4.txt'), object_hook=dict_to_sns)
    resT_a1b1e4 = json.load(open(path + 'results_trad_a1b1e4.txt'), object_hook=dict_to_sns)
    resN_a0b1 = json.load(open(path + 'results_ntrad_a0b1.txt'), object_hook=dict_to_sns)
    resT_a0b1 = json.load(open(path + 'results_trad_a0b1.txt'), object_hook=dict_to_sns)

    # resN_all = {'resN_a1b1': resN_a1b1,'resN_a1b1e4': resN_a1b1e4, 'resN_a0b1' : resN_a0b1}
    # resT_all = {'resT_a1b1': resT_a1b1, 'resT_a1b1e4' : resT_a1b1e4, 'resT_a0b1' : resT_a0b1}
    resN_all = {'resN_a1b1': resN_a1b1, 'resN_a0b1' : resN_a0b1}
    resT_all = {'resT_a1b1': resT_a1b1, 'resT_a0b1' : resT_a0b1}

    resN = SimpleNamespace(**resN_all)
    resT = SimpleNamespace(**resT_all)

    operN = []
    satN = []
    psN = []
    nsN = []
    nelemsN = []
    conv_solN = []
    conv_funcN = []
    conv_adjN = []
    stencilN = []
    cond_numN = []
    errsN = []
    errs_funcN = []
    probN = []
    norm_condN = []
    cond_numT_all = []

    # unpack dictionary and calculate convergence rates for operators with element-type refinement
    for k1 in resN.__dict__:
        case = resN.__dict__[k1]
        for k2 in case.__dict__:
            prob = case.__dict__[k2]
            for k3 in prob.__dict__:
                oper = prob.__dict__[k3]
                for k4 in oper.__dict__:
                    sat = oper.__dict__[k4]
                    for k5 in sat.__dict__:
                        ps = sat.__dict__[k5]
                        for k6 in ps.__dict__:
                            result = ps.__dict__[k6]
                            for k7 in result.__dict__.copy():
                                if (k7 == 'errs' or k7 == 'errs_func' or k7 == 'errs_adj') and k1 != 'resN_a1b1e4':
                                    err = np.asarray(result.__dict__[k7])
                                    nelems = np.asarray(result.nelems)
                                    xr = 1
                                    xl = 0
                                    hs = (xr - xl) / (np.asarray(nelems))
                                    slope = np.log2(err[0:-1]/err[1:])
                                    if k7 != 'errs_func':
                                        dx1 = slope[(slope <= result.p+3) & (slope >= result.p)]
                                        if len(dx1) != 0:
                                            indx1 = np.where(slope == dx1[0])[0][0] + 2
                                        else:
                                            indx1 = 0
                                        dx2 = slope[slope >= result.p-0.5]
                                        indx2 = np.where(slope == dx2[-1])[-1][-1] + 2
                                    else:
                                        indx1 = (np.where(slope <= 2*result.p + 3))[0][0] + 2
                                        if result.p > 2:
                                            indx2 = (np.where(slope >= result.p+0.5))[-1][-1] + 2
                                        elif result.p == 2:
                                            indx2 = (np.where(slope >= result.p + 0.5))[-1][-1] + 2
                                        else:
                                            indx2 = (np.where(slope >= result.p))[-1][-1] + 2

                                    conv = np.abs(np.polyfit(np.log10(hs[indx1:indx2]), np.log10(err[indx1:indx2]), 1)[0])

                                if k7 == 'errs':
                                    result.__dict__.update({'conv_sol': conv})
                                    conv_solN.append(conv)
                                elif k7 == 'errs_adj':
                                    result.__dict__.update({'conv_adj': conv})
                                    conv_adjN.append(conv)
                                elif k7 == 'errs_func':
                                    result.__dict__.update({'conv_func': conv})
                                    conv_funcN.append(conv)

                                    probN.append(k2)
                                    operN.append(result.quad_type)
                                    satN.append(result.flux_type_vis)
                                    psN.append(result.p)
                                    stencilN.append(k6)
                                    cond_numN.append(result.cond_num)
                                    errsN.append(result.errs)
                                    errs_funcN.append(result.errs_func)
                                    nsN.append(result.ns)
                                    nelemsN.append(result.nelems)

                                for y1 in resT.__dict__:
                                    caseT = resT.__dict__[y1]
                                    for y2 in caseT.__dict__:
                                        probT = caseT.__dict__[y2]
                                        for y3 in probT.__dict__:
                                            operT = probT.__dict__[y3]
                                            for y4 in operT.__dict__:
                                                psT = operT.__dict__[y4]
                                                for y6 in psT.__dict__:
                                                    resultT = psT.__dict__[y6]
                                                    for y7 in resultT.__dict__.copy():
                                                        if k7 == 'cond_num' and y7 == 'cond_num' and y1[4:]==k1[4:] and y2==k2 and y3==k3 and y4==k5 and y6==k6 and k1 != 'resN_a1b1e4':
                                                            norm_condN.append(np.asarray(result.cond_num)/np.asarray(resultT.cond_num))
                                                            cond_numT_all.append(resultT.cond_num)


    resN_list = [probN, operN, satN, psN, stencilN, conv_solN, conv_adjN, conv_funcN, nsN, nelemsN, errsN, errs_funcN, cond_numN, cond_numT_all, norm_condN]

    with open('resN_list.csv', 'w') as f:
        fc = csv.writer(f, lineterminator='\n')
        fc.writerows(resN_list)

    operT= []
    psT = []
    conv_solT = []
    conv_funcT = []
    conv_adjT = []
    stencilT = []
    nsT = []
    errsT = []
    errs_funcT = []
    cond_numT = []
    probT = []

    # unpack dictionary and calculate convergence rates for operators with Traditional refinement
    for k1 in resT.__dict__:
        case = resT.__dict__[k1]
        for k2 in case.__dict__:
            prob = case.__dict__[k2]
            for k3 in prob.__dict__:
                oper = prob.__dict__[k3]
                for k4 in oper.__dict__:
                    ps = oper.__dict__[k4]
                    for k6 in ps.__dict__:
                        result = ps.__dict__[k6]
                        for k7 in result.__dict__.copy():
                            if (k7 == 'errs' or k7 == 'errs_func' or k7 == 'errs_adj') and k1 != 'resT_a1b1e4':
                                err = np.asarray(result.__dict__[k7])
                                dofs = np.asarray(result.nelems) * np.asarray(result.ns)
                                xr = 1
                                xl = 0
                                hs = (xr - xl) / (np.asarray(dofs))
                                slope = np.log2(err[0:-1] / err[1:])
                                if k7 != 'errs_func':
                                    dx1 = slope[(slope <= result.p + 3) & (slope >= result.p)]
                                    if len(dx1) != 0:
                                        indx1 = np.where(slope == dx1[0])[0][0] + 2
                                    else:
                                        indx1 = 0
                                    dx2 = slope[slope >= result.p - 0.5]
                                    indx2 = np.where(slope == dx2[-1])[-1][-1] + 2
                                else:
                                    indx1 = (np.where(slope <= 2 * result.p + 3))[0][0] + 2
                                    if result.p > 2:
                                        indx2 = (np.where(slope >= result.p + 0.5))[-1][-1] + 2
                                    elif result.p == 2:
                                        indx2 = (np.where(slope >= result.p + 0.5))[-1][-1] + 2
                                    else:
                                        indx2 = (np.where(slope >= result.p))[-1][-1] + 2

                                conv = np.abs(np.polyfit(np.log10(hs[indx1:indx2]), np.log10(err[indx1:indx2]), 1)[0])

                                if k7 == 'errs':
                                    result.__dict__.update({'conv_sol': conv})
                                    conv_solT.append(conv)
                                elif k7 == 'errs_adj':
                                    result.__dict__.update({'conv_adj': conv})
                                    conv_adjT.append(conv)
                                elif k7 == 'errs_func':
                                    result.__dict__.update({'conv_func': conv})
                                    conv_funcT.append(conv)

                                    probT.append(k2)
                                    operT.append(result.quad_type)
                                    psT.append(result.p)
                                    stencilT.append(k6)
                                    cond_numT.append(result.cond_num)
                                    errsT.append(result.errs)
                                    errs_funcT.append(result.errs_func)
                                    nsT.append(result.ns)

    resT_list = [probT, operT, psT, stencilT, conv_solT, conv_adjT, conv_funcT, nsT, errsT, errs_funcT, cond_numT]
    with open('resT_list.csv', 'w') as f:
        fc = csv.writer(f, lineterminator='\n')
        fc.writerows(resT_list)

    # plots
    plot_cond(resN_list, resT_list)
    plot_conv(resN_list, resT_list)

    return

analyze_results()


