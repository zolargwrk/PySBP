import numpy as np
import quadpy
import warnings
import math


class CubatureRules:
    """Provides cubature rules for line, triangle, and tetrahedral

    Attributes:
        p : int     -- maximum degree of polynomial for which the cubature rule approximates exactly
        quad_type: str
            In 1D -- 'LGL', 'LG'

    Methods:
        quad_line_volume()
        quad_line_face()
    """

    def __init__(self, p, quad_type):
        self.p = p
        self.quad_type = quad_type

    def quad_line_volume(self, **kwargs):
        """Returns the nodal location and weights associated with quadrature rules in 1D
            **kwargs: n : int     -- number of degrees of freedom"""

        if len(kwargs) > 0:
            n = kwargs.values()
        else:
            n = self.p + 1

        if self.quad_type == 'LG':
            if n != self.p + 1:
                warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
            quadrule = quadpy.line_segment.gauss_legendre(n)
            xq = quadrule.points.reshape((n, 1))
            wq = quadrule.weights.reshape((n, 1))

        elif self.quad_type == 'LGL':
            if n != self.p + 1:
                warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
            quadrule = quadpy.line_segment.gauss_lobatto(n)
            xq = quadrule.points.reshape((n, 1))
            wq = quadrule.weights.reshape((n, 1))

        else:
            raise NotImplementedError
        return xq, wq

    def quad_line_face(self):
        xqf = np.array([-1, 1])
        wqf = 1.0
        return xqf, wqf

# quad = CubatureRules(5, 'LG')
# xp, xw = quad.quad_line_volume()
# print(xp)
# print(xw)