"""Matrix constructor for development of 1D SBP operators"""
import numpy as np
import sympy as sym
import quadpy
import sys
sym.init_printing()


def mesh_1d(xl, xr, m, **kwargs):
    """Creates equidistance nodes for a given interval
    Inputs: xl - left end point
            xr - right end point
            m  - number of nodes
    Optional inputs: scheme = 'CC'  - Clenshaw-Curtis
                     scheme = 'LGL' - Gauss-Lagendre
                     scheme = 'LG'  - Gauss-Lobatto
                     scheme = 'LGR' - Gauss-Radau
                     scheme = 'NC'  - Newton-Cotes
    Output: x - vector of equidistance nodes
    """

    if 'CC' in list(kwargs.values()):
        scheme = quadpy.line_segment.clenshaw_curtis(m)
        pts = scheme.points
        x = 1 / 2 * (xl * (1 - pts) + xr * (1 + pts))    # linear mapping to the domain
    elif 'LG' in list(kwargs.values()):
        scheme = quadpy.line_segment.gauss_legendre(m)
        pts = scheme.points
        x = 1/2*(xl*(1 - pts) + xr*(1 + pts))
    elif 'LGL' in list(kwargs.values()):
        scheme = quadpy.line_segment.gauss_lobatto(m)
        pts = scheme.points
        x = 1/2*(xl*(1 - pts) + xr*(1 + pts))
    elif 'NC' in list(kwargs.values()):
        scheme = quadpy.line_segment.newton_cotes_closed(m)
        pts = scheme.points
        x = 1 / 2 * (xl * (1 - pts) + xr * (1 + pts))
    elif 'LGR' in list(kwargs.values()):
        scheme = quadpy.line_segment.gauss_radau(m)
        pts = scheme.points
        x = 1 / 2 * (xl * (1 - pts) + xr * (1 + pts))
    else:
        x = np.linspace(xl, xr, m)
    return x


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


def construct_h_mat(p, m,  xl, xr):
    """Construct the diagonal H norm matrix
    Inputs: p - degree of operator
            m - number of nodes
            xl, xr - left and right end points
    Output: h_mat  - diagonal H norm matrix
            all_vars - variables in the H matrix"""

    # number of boundary nodes (see paper by Klarman and Albin)
    bns = {'p1': 2, 'p2': 4, 'p3': 6, 'p4': 8, 'p5': 11, 'p6': 14, 'p7': 19, 'p8': 23}
    if p >= 9:
        sys.exit('Only operators up to degree p = 8 are supported.')
    bn = list(bns.values())[p-1]
    all_var = list()

    dx = sym.Rational((xr - xl) / (m - 1))
    h_mat = dx*sym.eye(m, m)

    # set variable at the boundaries of the H matrix
    for i in range(0, bn):
        h_mat[i, i] = sym.Symbol('h%d%d' % (i + 1, i + 1))
        all_var.append(h_mat[i, i])

    for i in range(m - bn, m):
        h_mat[i, i] = sym.Symbol('h%d%d' % (m - i, m - i))

    return h_mat, all_var


def construct_s_int(p):
    """Construct interior operator for the S matrix
    i.e., center difference operators of degree 2p
    Inputs: p - degree of operator
    Output: s_int - interior operator"""

    s_int = sym.zeros(1, 2 * p + 1)
    k = p
    for i in range(0, p):
        s_int[0, 2*p-i] = (-1)**(k+1)*sym.factorial(p)**2/(k*sym.factorial(p+k)*sym.factorial(p-k))
        s_int[0, i] = -1*s_int[0, 2*p-i]
        k = k-1
    return s_int


def construct_s_mat(p, m, s_int):
    """Construct the S matrix: S = Q+E
    Inputs: p - degree of operator
            m - number of nodes
            s_int - interior operator
    Output: s_mat - the antisymmetric matrix
            all_var - variables in the S matrix"""

    # number of boundary nodes (see paper by Klarman and Albin)
    bns = {'p1': 2, 'p2': 4, 'p3': 6, 'p4': 8, 'p5': 11, 'p6': 14, 'p7': 19, 'p8': 23}
    if p >= 9:
        sys.exit('Only operators up to degree p = 8 are supported.')
    bn = list(bns.values())[p-1]
    all_var = list()

    s_mat = sym.zeros(m, m)  # S matrix
    for i in range(p, m - p):
        s_mat[i, (i - p):(i + p + 1)] = s_int[0, 0:(2 * p + 1)]

    for i in range(0, bn):
        for j in range(0, bn):
            if i == j:
                s_mat[i, j] = 0
            else:
                s_mat[i, j] = -sym.Symbol('s%d%d' % (j + 1, i + 1))
                s_mat[j, i] = sym.Symbol('s%d%d' % (j + 1, i + 1))
                if j < i:
                    all_var.append(s_mat[j, i])

    for i in range(0, bn):
        for j in range(0, m):
            s_mat[m - i - 1, m - j - 1] = -s_mat[i, j]
    return s_mat, all_var


def construct_acc_list(p, m, x, h_mat, q_mat):
    """Construct the accuracy equations to solve for unknowns
    Inputs: p - degree of operator
            m - number of degrees of freedom
            x - 1D mesh
            h_mat - the H norm matrix
            q_mat - the Q matrix, Q = S+E
    Output: acc_list - the list of the accuracy equations"""

    acc_mat = sym.zeros(m, p+1)
    for i in range(0, p+1):
        temp = np.dot(q_mat, xp_vec(m, i, x)) - i*np.dot(h_mat, xp_vec(m, i-1, x))
        acc_mat[:, i] = temp[:, 0]

    acc_list = list()
    for i in range(0, m):
        for j in range(0, p+1):
            acc_list.append(acc_mat[i, j])

    return acc_list


def construct_c_mat(p, m, n):
    """Construct the C matrix for the second derivative operator
    Inputs: p - degree of operator
            m - number of degrees of freedom
            n - identifier of the C matrix
    Output: c_mat - the C matrix
            all_var - variables in the C matrix"""

    # number of boundary nodes (see paper by Klarman and Albin)
    bns = {'p1': 2, 'p2': 4, 'p3': 6, 'p4': 8, 'p5': 11, 'p6': 14, 'p7': 19, 'p8': 23}
    if p >= 9:
        sys.exit('Only operators up to degree p = 8 are supported.')
    bn = list(bns.values())[p-1]
    all_var = list()

    c_mat = sym.eye(m, m)
    for i in range(0, bn):
        c_mat[i, i] = sym.Symbol('c%d%d' % (n, i + 1))
        all_var.append(c_mat[i, i])

    for i in range(m - bn, m):
        c_mat[i, i] = sym.Symbol('c%d%d' % (n, m - i))

    return c_mat, all_var


def construct_d_mat(p, m, s_int, n):
    """Construct the derivative operator
    Inputs: p - degree of operator
            m - number of nodes
            s_int - interior operator
            n - identifier
    Output: d_mat - the first derivative matrix D1
            all_var - variables in the S matrix"""

    # number of boundary nodes (see paper by Klarman and Albin)
    bns = {'p1': 2, 'p2': 4, 'p3': 6, 'p4': 8, 'p5': 11, 'p6': 14, 'p7': 19, 'p8': 23}
    if p >= 9:
        sys.exit('Only operators up to degree p = 8 are supported.')
    bn = list(bns.values())[p-1]
    bn2 = bn + p

    all_var = list()
    d_mat = sym.zeros(m, m)
    k = int((len(s_int)-1)/2)
    for i in range(bn, m - bn):
        d_mat[i, (i - k):(i + k+1)] = np.asmatrix(s_int)

    for i in range(0, bn):
        for j in range(0, bn2):
            d_mat[i, j] = sym.Symbol('d%d%d%d' % (n, i + 1, j + 1))
            all_var.append(d_mat[i, j])

    for i in range(0, bn):
        for j in range(0, m):
            if n % 2 == 0:  # check if n is even
                d_mat[m - i - 1, m - j - 1] = d_mat[i, j]
            else:
                d_mat[m - i - 1, m - j - 1] = -d_mat[i, j]
    return d_mat, all_var


def construct_db_mat(p, m, **kwargs):
    """Construct the S matrix: S = Q+E
    Inputs: p - degree of operator
            m - number of nodes
            tl - left projection vector
            kwargs: scheme = 'LG' - Legendre-Gauss quadrature
                    leave blank for schemes
    Output: db_mat - the normal first derivative matrix D1b at the boundary
            all_var - variables in the S matrix"""

    # number of boundary nodes (see paper by Klarman and Albin)
    bns = {'p1': 2, 'p2': 4, 'p3': 6, 'p4': 8, 'p5': 11, 'p6': 14, 'p7': 19, 'p8': 23}
    if p >= 9:
        sys.exit('Only operators up to degree p = 8 are supported.')
    bn = list(bns.values())[p-1]
    bn2 = bn + p

    all_var = list()
    db_mat = sym.zeros(m, m)
    if 'LG' in list(kwargs.values()):
        for i in range(0, bn):
            for j in range(0, bn2):
                db_mat[i, j] = sym.Symbol('db%d%d' % (i + 1, j + 1))
                all_var.append(db_mat[i, j])

        for i in range(0, bn):
            for j in range(0, m):
                db_mat[m - i - 1, m - j - 1] = -db_mat[i, j]
    else:
        for j in range(0, bn2):
            db_mat[0, j] = sym.Symbol('db%d%d' % (1, j + 1))
            all_var.append(db_mat[0, j])

        for j in range(0, m):
            db_mat[m - 1, m - j - 1] = -db_mat[0, j]

    return db_mat, all_var


def lagrange(p, x0, x):
    """Evaluates the i-th Lagrange polynomial at x0 based on grid data x
    Inputs: x0 - point at which we want to evaluate the Lagrange polynomial
            p  - degree of the Lagrange polynomial
            x  - 1D mesh
    Output: y  - Lagrange polynomial value at point x0
    """
    m = len(x)
    y = 1.
    for j in range(0, m):
        if p != j:
            y *= (x0-x[j])/(x[p]-x[j])
    return y


def construct_projectors_1d(xl, xr, x, **kwargs):
    """Construct the boundary projection matrices
    Inputs: p   - degree of operator
            xl  - left end point of domain
            xr  - right end point of domain
            x   - 1D mesh
            kwargs: scheme = 'LG'  - Legendre-Gauss
                    scheme = 'LGR' - Legendre-Gauss-Radau
                    leave blank for other schemes
    Output: tl - left projection matrix
            tr - right projection matrix"""

    m = len(x)
    tl = np.zeros((m, 1))
    tr = np.zeros((m, 1))

    if 'LG' or 'LGR' in list(kwargs.values()):
        for i in range(0, m):
            tl[i] = lagrange(i, xl, x)
            tr[i] = lagrange(i, xr, x)
    else:
        tl[1] = 1
        tr[m] = 1

    return tl, tr

# db_mat_all = construct_db_mat(1, 10, opt = 'LG')
# sym.pprint(db_mat_all[0])