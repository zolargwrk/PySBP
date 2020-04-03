import numpy as np
import sympy as sym


def operator_test_1d(p, x, h_mat, d_mat, **kwargs):
    """ Test for accuracy of first and second derivative operators and whether the norm matrix is positive definite
        Inputs: p - degree of operator
                x - 1D mesh
                h_mat - norm matrix
                d_mat - first derivative operator
                **kwargs - der = 'd2p': second derivative operator
        Output: returns message with the maximum error"""

    m = len(x)
    # check if the norm matrix has only positive weights
    h_diag = list()
    for i in range(0, m):
        h_diag.append(h_mat[i, i])
    h_diag = np.asarray(h_diag)
    h_min = h_diag.min()

    # test for accuracy
    if len(kwargs.values()) != 0:
        d2p = list(kwargs.values())[0]
        errs = np.zeros((p + 2, 1))

        print('Maximum error for the 2nd derivative of degree p =', p)
        for i in range(0, p + 2):
            err = np.squeeze(np.asarray(d2p @ xp_vec(i, x))) - i * (i - 1) * np.squeeze(np.asarray(xp_vec(i - 2, x)))
            # print(err)
            errs[i, 0] = (abs(err)).max()
            print('err_max for p =', i, ':  ', errs[i, 0])
        err_max = errs.max()

    else:
        errs = np.zeros((p + 1, 1))
        print('Maximum error for the 1st derivative of degree p =', p)
        for pp in range(0, p + 1):
            err = d_mat @ xp_vec(pp, x) - pp * xp_vec(pp - 1, x)
            # print(err)
            errs[pp, 0] = err.max()
            print('err_max for p =', pp, ':  ', errs[pp, 0])
        err_max = errs.max()
        if h_min <= 0 or err_max >= 1e-6:
            print('Fail: H norm matrix has entries <= 0 or operator has error greater than 1e-6.'
                  ' \n max_err = ', err_max, '\n H_min = ', h_min)


def xp_vec(k, x, **kwargs):
    """Returns pointwise kth power of the 1D mesh vector x
    Inputs: k - power
            x - the discretization matrix x
            kwargs:  dtype = 'symbolic' for symbolic computation
                     dtype = 'float' or leave option for numerical computation
    Output: xp - kth power of x
    """
    m = len(x)
    if 'symbolic' in list(kwargs.values()):
        xp = sym.zeros(m, 1)
        for i in range(0, m):
            if k < 0:
                xp[i] = 0
            else:
                xp[i] = sym.Rational(x[i] ** k)
    else:
        xp = np.ndarray((m, 1))
        for i in range(0, m):
            if k < 0:
                xp[i] = 0
            else:
                xp[i] = x[i] ** k
    return xp