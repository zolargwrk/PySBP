import numpy as np
import quadpy
import warnings
import math


class CubatureRules:
    """Provides cubature rules for line, triangle, and tetrahedral

    Attributes:
        p (int): maximum degree of polynomial for which the cubature rule approximates exactly
        quad_type (str):  In 1D -- 'LGL', 'LG'

    Methods:
        quad_line_volume()
        quad_line_face()
    """

    @staticmethod
    def quad_line_volume(p, quad_type, n=0):
        """Returns the nodal location and weights associated with quadrature rules in 1D
            **kwargs: n (int):     -- number of degrees of freedom"""

        if n == 0:
            n = p+1

        if quad_type == 'LG':
            if n != p + 1:
                warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
            quadrule = quadpy.line_segment.gauss_legendre(n)
            xq = quadrule.points.reshape((n, 1))
            wq = quadrule.weights.reshape((n, 1))

        elif quad_type == 'LGL':
            if n != p + 1:
                warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
            quadrule = quadpy.line_segment.gauss_lobatto(n)
            xq = quadrule.points.reshape((n, 1))
            wq = quadrule.weights.reshape((n, 1))

        else:
            raise NotImplementedError
        return xq, wq

    @staticmethod
    def quad_line_face(p, quad_type):
        xqf = np.array([-1, 1])
        wqf = 1.0
        return xqf, wqf

    @staticmethod
    def cub_tri_volume(p, sbp_family, p_cub=0):
        """A module to obtain the cubature points and weights for SBP-Gamma, SBP-Omega and SBP-DiagE operators
        Args:
            p(int): degree of the cubature rule
            sbp_family (str): the type of cubature rule, "SBP-Gamma", "SBP-Omega", or "SBP-DiagE"
            p_cub (int, optional): the degree of the cubature rule which is set to 2p-1

        Returns:
            r(array): the x coordinates of the cubature points on the reference right triangle
            s(array): the y coordinates of the cubature points on the reference right triangle
            w(array): the weight of the cubature rule at the cubature points

        """
        if p_cub == 0:
            p_cub = 2*p - 1

        sbp_family = str.lower(sbp_family)
        if sbp_family == "gamma":
            if p == 1:
                if p_cub == 1:
                    r = np.array([-1.0,  1.0, -1.0])
                    s = np.array([-1.0, -1.0,  1.0])
                    w = np.array([0.6666666666666666,  0.6666666666666666,  0.6666666666666666])
                elif p_cub == 2:
                    r = np.array([0.333333333333333, 0, 1, 0])
                    s = np.array([0.333333333333333, 0, 0, 1])
                    w = np.array([0.375, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=1 and p_cub=2 are available.")

            elif p == 2:
                if p_cub == 3:
                    r = np.array([-1.0, 1.0, -1.0, 0.0, 0.0, -1.0, -0.3333333333333333])
                    s = np.array([-1.0, -1.0,  1.0, -1.0, 0.0,  0.0, -0.3333333333333333])
                    w = np.array([0.09999999999999999,  0.09999999999999999,  0.09999999999999999,  0.26666666666666666, 0.26666666666666666,  0.26666666666666666,  0.9000000000000002])
                elif p_cub == 4:
                    r = np.array([0.188580484696445, 0.6228390306071099, 0.188580484696445, 0, 1, 0, 0.5, 0, 0.5])
                    s = np.array([0.188580484696445, 0.188580484696445, 0.6228390306071099, 0, 0, 1, 0.5, 0.5, 0])
                    w = np.array([0.1254088495595664, 0.1254088495595664, 0.1254088495595664, 0.01027006767296668, 0.01027006767296668, 0.01027006767296668, 0.03098774943413357, 0.03098774943413357, 0.03098774943413357])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=1 and p_cub=2 are available.")

            elif p == 3:
                if p_cub == 5:
                    r = np.array([-1.0,  1.0, -1.0, -0.4130608881819197, 0.4130608881819197,  0.4130608881819197, -0.4130608881819197, -1.0, -1.0,  -0.5853096486728182,  0.17061929734563638, -0.5853096486728182])
                    s = np.array([-1.0, -1.0,  1.0, -1.0, -1.0, -0.4130608881819197,  0.4130608881819197,  0.4130608881819197, -0.4130608881819197,  -0.5853096486728182, -0.5853096486728182,  0.17061929734563636])
                    w = np.array([0.02974582604964118,  0.02974582604964118,  0.02974582604964118,  0.09768336246810204, 0.09768336246810204,  0.09768336246810204,  0.09768336246810204,  0.09768336246810204, 0.09768336246810204,   0.4415541156808217,  0.4415541156808217,  0.4415541156808217])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=1 and p_cub=2 are available.")

            elif p == 4:
                if p_cub == 7:
                    r = np.array([-1.0,  1.0, -1.0, -0.5773502691896257,  0.0,  0.5773502691896257, 0.5773502691896257, 0.0, -0.5773502691896257, -1.0, -1.0, -1.0, -0.7384168123405102, -0.1504720765483788,  0.4768336246810203, -0.1504720765483788, -0.7384168123405102, -0.6990558469032424])
                    s = np.array([-1.0, -1.0,  1.0, -1.0, -1.0, -1.0, -0.5773502691896257, 0.0, 0.5773502691896257, 0.5773502691896257,  0.0, -0.5773502691896257, -0.7384168123405102, -0.6990558469032424, -0.7384168123405102, -0.15047207654837885,  0.47683362468102025, -0.15047207654837885])
                    w = np.array([0.012698412698412695, 0.012698412698412695, 0.012698412698412695, 0.04285714285714284, 0.05079365079365077,  0.04285714285714284,  0.04285714285714284, 0.05079365079365077,  0.04285714285714284,  0.04285714285714284,  0.05079365079365077,  0.04285714285714284,  0.2023354595827503,  0.3151248578775673,  0.2023354595827503,  0.3151248578775673,  0.2023354595827503,  0.3151248578775673])
                else:
                    raise ValueError("Only cubature rules of degree p_cub=1 and p_cub=2 are available.")

            elif p >= 5:
                raise ValueError("Only cubature rules of degree up to p=4 are available for the SBP-Gamma family.")

        r = r.reshape(r.shape[0], 1)
        s = s.reshape(s.shape[0], 1)
        w = w.reshape(w.shape[0], 1)
        cub_vert = np.array([[-1,-1], [1, -1], [-1, 1]])

        return {'r': r, 's': s, 'w': w, 'cub_vert': cub_vert}


# quad = CubatureRules(5, 'LG')
# xp, xw = quad.quad_line_volume()
# print(xp)
# print(xw)

# cub = CubatureRules.cub_tri_volume(1, "SBP-gamma")
# print(cub['s'])