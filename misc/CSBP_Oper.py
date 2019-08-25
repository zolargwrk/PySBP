# routines to solve the coefficients in the norm matrix, H, and the Q matrix of the Classical Summation-By-Parts,
# and the order-matched and compatible 2nd derivative operator of degree 1 (unfortunately the symbolic toolbox in
# python couldn't solve for degrees more than 1; hence, solutions for higher degree are obtained using Maple 2017)

import numpy as np
import scipy as sp
import sys
from scipy.optimize import minimize
import sympy as sym
from construct_matrices_sbp_1d import *
from sympy.solvers.solveset import nonlinsolve
sym.init_printing()


def hqd_csbp(p):
    m = 4*p + 5    # size of matrix to construct operator
    # if p <= 4:
    #     bn = 2*p    # number of boundary nodes
    if p == 1:
        bn = 2
    elif p == 2:
        bn = 4
    elif p == 3:
        bn = 6
    elif p == 4:
        bn = 8
    elif p == 5:
        bn = 11         # size of bn is based on the paper by Klarmnn and Albin
    elif p == 6:
        bn = 14
    elif p == 7:
        bn = 19
    elif p == 8:
        bn = 23
    elif p >= 9:
        sys.exit('Only operators up to degree p = 8 are supported.')

    # dx = sym.Rational((xr-xl)/(m-1))          # mesh size
    x = np.linspace(1, m, m)                    # 1D mesh
    h_mat = sym.eye(m, m)                       # H norm matrix
    all_var = list()                            # list of all variables

    # set variable at the boundaries of the H matrix
    for i in range(0, bn):
        h_mat[i, i] = sym.Symbol('h%d%d' % (i+1, i+1))
        all_var.append(h_mat[i, i])

    for i in range(m-bn, m):
        h_mat[i, i] = sym.Symbol('h%d%d' % (m-i, m-i))

    # construct interior operator for the S matrix
    s_int = sym.zeros(1, 2 * p + 1)
    k = p
    for i in range(0, p):
        s_int[0, 2*p-i] = (-1)**(k+1)*sym.factorial(p)**2/(k*sym.factorial(p+k)*sym.factorial(p-k))
        s_int[0, i] = -1*s_int[0, 2*p-i]
        k = k-1
        # note that the sym.factorial keeps the fraction to be symbolic (rational) instead of float type

    # construct the S matrix
    s_mat = sym.zeros(m, m)                         # S matrix
    for i in range(p, m-p):
        s_mat[i, (i-p):(i+p+1)] = s_int[0, 0:(2*p+1)]

    for i in range(0, bn):
        for j in range(0, bn):
            if i == j:
                s_mat[i, j] = 0
            else:
                s_mat[i, j] = -sym.Symbol('s%d%d' % (j+1, i+1))
                s_mat[j, i] = sym.Symbol('s%d%d' % (j+1, i+1))
                if j < i:
                    all_var.append(s_mat[j, i])

    for i in range(0, bn):
        for j in range(0, m):
            s_mat[m-i-1, m-j-1] = -s_mat[i, j]

    # construct the E matrix
    e_mat = sym.zeros(m, m)
    e_mat[0, 0] = -1
    e_mat[m-1, m-1] = 1
    # proj = construct_projectors_1d(1, m, x, scheme = 'LG')

    # construct the Q matrix
    q_mat = s_mat + sym.Rational(1/2)*e_mat

    # construct the accuracy equations
    acc_mat = sym.zeros(m, p+1)
    for i in range(0, p+1):
        temp = np.dot(q_mat, xp_vec(i, x, dtype='symbolic')) - i*np.dot(h_mat, xp_vec(i-1, x, dtype='symbolic'))
        acc_mat[:, i] = temp[:, 0]

    acc_list = list()
    for i in range(0, m):
        for j in range(0, p+1):
            acc_list.append(acc_mat[i, j])

    sol = sym.solve(acc_list)
    h_mat = h_mat.subs(sol)
    q_mat = q_mat.subs(sol)
    d_mat = h_mat.inv() * q_mat

    # ================================================================================================================ #
    # optimization for operators with free variables
    # ================================================================================================================ #

    # calculate error
    err1 = d_mat*xp_vec(p+1, x, dtype='symbolic') - (p + 1)*xp_vec(p, x, dtype='symbolic')
    err2 = err1.transpose()*h_mat*err1

    # get free variables
    free_var = set(all_var).difference(set(list(sol.keys())))

    if len(free_var) > 0 and p <= 4:
        # determine the gradient of the error with respect to the free variables
        free_var_list = sym.Array(list(free_var))
        grad_err = sym.diff(err2[0, 0], free_var_list)
        kkt_eqs = grad_err   # there is no Lagrange multiplier since problem is unconstrained

        # solve for values of free variables that minimize the truncation error
        sol2 = sym.solve(kkt_eqs)
        print(type(kkt_eqs))
        # if there are free variables after optimization, set them to zero
        free_var2 = set(free_var_list).difference(set(list(sol2.keys())))
        free_var2 = dict.fromkeys(free_var2, 0)

        # substitute values of free variables into the Q matrix
        q_mat = q_mat.subs(sol2)
        q_mat = q_mat.subs(free_var2)
        d_mat = h_mat.inv() * q_mat

    elif len(free_var) > 0 and p > 4:
        # for p > 5 set free variables such that H is positive definite (PD) -- note: optimization takes long time
        # optimization for p > 4 operators is certainly necessary since the error introduced is large
        eqs1 = list()
        for i in range(0, m):
            eqs1.append(h_mat[i, i])
        eqs1 = sym.Array(eqs1)

        # write the system of equation in matrix form
        free_var_list = sym.Array(list(free_var))
        eqs1_mat = sym.linear_eq_to_matrix(eqs1, free_var_list)
        a_mat = (list(eqs1_mat)[0]).evalf()
        b_mat = (list(eqs1_mat)[1]).evalf()

        # solve the system of inequalities as a minimization problem using "linprog" functionality in scipy
        a_ub = np.array(-a_mat)
        b_ub = np.array(-b_mat) - 0.005  # subtract small number to avoid problem with inverse of H due to a zero entry,
        # i.e., we require that Ax + b >= e instead of Ax + b >= 0, where e is some small number (used 0.005 for p=6)
        c = np.zeros(len(free_var_list))
        sol3 = sp.optimize.linprog(c, a_ub, b_ub, options={"maxiter": 100})
        sol3 = list(sol3.x)
        free_var3 = dict(zip(free_var_list, sol3))

        # substitute solution of free variables in to Q and H matrices
        q_mat = q_mat.subs(free_var3)
        h_mat = h_mat.subs(free_var3)
        d_mat = h_mat.inv() * q_mat

    # ================================================================================================================ #
    #                                       second derivative operators
    # ================================================================================================================ #

    if p == 1:
        d2_int = [1, -2, 1]
        a1 = sym.Rational(-1 / 4)
        d2_mat_all = construct_d_mat(p, m, d2_int, 2)
        d2_mat = d2_mat_all[0]
        var_d2 = d2_mat_all[1]
        c2_mat_all = construct_c_mat(p, m, 2)
        c2_mat = c2_mat_all[0]
        var_c2 = c2_mat_all[1]

        # solve for the d2_mat such that it is first order accurate approximation of the second derivative
        # eqd20 = np.dot(d2_mat, xp_vec(0, x, dtype='symbolic'))
        # eqd21 = np.dot(d2_mat, xp_vec(1, x, dtype='symbolic'))
        # eqd22 = np.dot(d2_mat, xp_vec(2, x, dtype='symbolic')) - 2*xp_vec(0, x, dtype='symbolic')
        eqd2 = list()
        for i in range(0, p+2):
            eqd2.append(np.dot(d2_mat, xp_vec(i, x, dtype='symbolic')) - sym.factorial(i)*xp_vec(i-2, x, dtype='symbolic'))
        # print(eqd2[2])
        eqsd2 = list(eqd2)
        # eqsd2 = list(eqd20[:, 0]) + list(eqd21[:, 0]) + list(eqd22[:, 0])
        # print(eqsd2)
        sol_eqsd2 = sym.solve(eqsd2)
        free_var_d2 = set(var_d2).difference(set(list(sol_eqsd2.keys())))

        # substitute solution to the d2_mat
        d2_mat = d2_mat.subs(sol_eqsd2)

        # construct the db_mat
        db_mat_all = construct_db_mat(p, m)
        db_mat = db_mat_all[0]
        var_db = db_mat_all[1]

        eqdb = list()
        eqsdb = list()
        for i in range(0, p + 2):
            eqdb.append(np.dot(db_mat, xp_vec(i, x, dtype='symbolic')) - i * xp_vec(i - 1, x, dtype='symbolic'))
            eqsdb.append(eqdb[i][0, 0])
            eqsdb.append(eqdb[i][m - 1, 0])
        eqsdb = list(eqsdb)
        sol_eqsdb = sym.solve(eqsdb)
        free_var_db = set(var_db).difference(set(list(sol_eqsdb.keys())))

        # substitute solution to the db_mat
        db_mat = db_mat.subs(sol_eqsdb)

        # construct the second derivative operator of degree p = 1
        d2p1 = h_mat.inv()@(-d_mat.transpose()@h_mat@d_mat + a1*d2_mat.transpose()@c2_mat@d2_mat + e_mat@db_mat)

        eqd2p1 = list()
        temp = list()
        for i in range(0, p + 2):
            eqd2p1.append(np.dot(d2p1, xp_vec(i, x, dtype='symbolic')) - i*(i-1) * xp_vec(i - 2, x, dtype='symbolic'))
            temp.append(list(eqd2p1[i][:, 0]))
        acc_list_d2p1 = list()
        for i in range(0, len(temp)):
            for j in range(0, len(temp[0])):
                acc_list_d2p1.append(temp[i][j])

        # collect all free variables together
        free_var_d2p1 = list(free_var_db) + list(free_var_d2) + list(var_c2)

        # solve using nonlinear equation solver
        sol_d2p1 = nonlinsolve(acc_list_d2p1, list(free_var_d2p1))
        sol_d2p1 = list(sol_d2p1.args[0])
        sol_d2p1 = dict(zip(list(free_var_d2p1), sol_d2p1))

        # identify free variables remaining after solving and set them to zero
        temp_list = list()
        for i in range(0, len(sol_d2p1)):
            if (list(sol_d2p1.keys())[i]) is (list(sol_d2p1.values())[i]):
                temp_list.append(list(sol_d2p1.values())[i])
        free_var2_d2p1 = dict.fromkeys(temp_list, 0)
        sol_d2p1_array = sym.Array(list(sol_d2p1.values()))
        free_var_values = sol_d2p1_array.subs(free_var2_d2p1)
        sol_d2p1 = dict(zip(sol_d2p1.keys(), list(free_var_values)))

        # substitute solution into second derivative operator
        d2p1 = d2p1.subs(sol_d2p1)
        d2_mat = d2_mat
        c2_mat = c2_mat.subs(sol_d2p1)
        db_mat = db_mat.subs(sol_d2p1)

        # check error
        errs = np.zeros((p+2, 1))
        for pp in range(0, p+2):
            err = np.dot(d2p1, xp_vec(pp, x)) - pp*(pp-1)*xp_vec(pp - 2, x)
            errs[pp, 0] = err.sum()
        err_max = errs.max()

    # ================================================================================================================ #
    # could not find solution for p = 2.

    elif p == 2:
        d3_int = [sym.Rational(-1/2), 1, 0, -1, sym.Rational(1/2)]
        d4_int = [1, -4, 6, -4, 1]
        a3 = sym.Rational(-1/18)
        a4 = sym.Rational(-1/48)
        d3_mat_all = construct_d_mat(p, m, d3_int, 3)
        d3_mat = d3_mat_all[0]
        var_d3 = d3_mat_all[1]
        d4_mat_all = construct_d_mat(p, m, d4_int, 4)
        d4_mat = d4_mat_all[0]
        var_d4 = d4_mat_all[1]
        c3_mat_all = construct_c_mat(p, m, 3)
        c3_mat = c3_mat_all[0]
        var_c3 = c3_mat_all[1]
        c4_mat_all = construct_c_mat(p, m, 4)
        c4_mat = c4_mat_all[0]
        var_c4 = c4_mat_all[1]

        eqd3 = list()
        for i in range(0, p + 2):
            eqd3.append(np.dot(d3_mat, xp_vec(i, x, dtype='symbolic')) - sym.factorial(i)*xp_vec(i - 3, x, dtype='symbolic'))
        eqsd3 = list(eqd3)

        eqd4 = list()
        for i in range(0, p + 3):
            eqd4.append(np.dot(d4_mat, xp_vec(i, x, dtype='symbolic')) - sym.factorial(i)*xp_vec(i - 4, x, dtype='symbolic'))
        eqsd4 = list(eqd4)
        # print(eqsd4)

        sol_eqsd3 = sym.solve(eqsd3)
        free_var_d3 = set(var_d3).difference(set(list(sol_eqsd3.keys())))
        sol_eqsd4 = sym.solve(eqsd4)
        free_var_d4 = set(var_d4).difference(set(list(sol_eqsd4.keys())))

        # substitute solution to the d3_mat
        d3_mat = d3_mat.subs(sol_eqsd3)
        d4_mat = d4_mat.subs(sol_eqsd4)
        print(free_var_d3)
        print(free_var_d4)
        # construct the db_mat
        db_mat_all = construct_db_mat(p, m)
        db_mat = db_mat_all[0]
        var_db = db_mat_all[1]

        # solve: the db matrix should be at least third order accurate approximation to the first derivative
        eqdb = list()
        eqsdb = list()
        for i in range(0, p + 2):
            eqdb.append(np.dot(db_mat, xp_vec(i, x, dtype='symbolic')) - i * xp_vec(i - 1, x, dtype='symbolic'))
            eqsdb.append(eqdb[i][0, 0])
            eqsdb.append(eqdb[i][m-1, 0])
        eqsdb = list(eqsdb)

        sol_eqsdb = sym.solve(eqsdb)
        free_var_db = set(var_db).difference(set(list(sol_eqsdb.keys())))
        print(sol_eqsdb)

        # substitute solution to the db_mat
        db_mat = db_mat.subs(sol_eqsdb)

        # construct the second derivative operator of degree p = 2
        d2p2 = h_mat.inv() @ (-d_mat.transpose() @ h_mat @ d_mat + a3 * d3_mat.transpose() @ c3_mat @ d3_mat
                              + a4 * d4_mat.transpose() @ c4_mat @ d4_mat + e_mat @ db_mat)

        # construct accuracy equations
        eqd2p2 = list()
        temp = list()
        for i in range(0, p + 2):
            eqd2p2.append(np.dot(d2p2, xp_vec(i, x, dtype='symbolic')) - i*(i-1) * xp_vec(i - 2, x, dtype='symbolic'))
            temp.append(list(eqd2p2[i][:, 0]))
        acc_list_d2p2 = list()
        for i in range(0, len(temp)):
            for j in range(0, len(temp[0])):
                acc_list_d2p2.append(temp[i][j])

        print(acc_list_d2p2)
        # collect all free variables together
        free_var_d2p2 = list(free_var_db) + list(free_var_d3) + list(free_var_d4) + list(var_c3) + list(var_c4)
        print(free_var_d2p2)
        # solve using nonlinear equation solver
        sol_d2p2 = nonlinsolve(acc_list_d2p2, list(free_var_d2p2))
        sol_d2p2 = list(sol_d2p2.args[0])
        sol_d2p2 = dict(zip(list(free_var_d2p2), sol_d2p2))

        # identify free variables remaining after solving and set them to zero
        temp_list = list()
        for i in range(0, len(sol_d2p2)):
            if (list(sol_d2p2.keys())[i]) is (list(sol_d2p2.values())[i]):
                temp_list.append(list(sol_d2p2.values())[i])
        free_var2_d2p2 = dict.fromkeys(temp_list, 0)
        sol_d2p2_array = sym.Array(list(sol_d2p2.values()))
        free_var_values = sol_d2p2_array.subs(free_var2_d2p2)
        sol_d2p2 = dict(zip(sol_d2p2.keys(), list(free_var_values)))

        # substitute solution into second derivative operator
        d2p2 = d2p2.subs(sol_d2p2)

        # check error
        errs = np.zeros((p + 2, 1))
        for pp in range(0, p + 2):
            err = np.dot(d2p2, xp_vec(pp, x)) - pp * (pp - 1) * xp_vec(pp - 2, x)
            errs[pp, 0] = err.sum()
        err_max = errs.max()

    return h_mat, q_mat, d_mat, d2p1


kk = hqd_csbp(1)

