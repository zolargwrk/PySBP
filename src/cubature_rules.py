import numpy as np
import quadpy
import warnings


class CubatureRules:
    """Provides cubature rules for line, triangle, and tetrahedral

    Attributes:
        d : int     -- dimension
        p : int     -- maximum degree of polynomial for which the cubature rule approximates exactly
        quad_type: str
            In 1D -- 'LGL', 'LG', 'CSBP', 'HGTL', 'HGT' (HGTL : Hybrid-Gauss-Trapezoidal-Lobatto)

    Methods:
        quad_line()
    """

    def __init__(self, d, p, quad_type):
        self.d = d
        self.p = p
        self.quad_type = quad_type

    def quad_line(self, n):
        """Returns the nodal location and weights associated with quadrature rules in 1D
            n : int     -- number of degrees of freedom"""
        if self.d == 1:
            if self.quad_type == 'LG':
                m = self.p + 1
                if n != m:
                    warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
                quadrule = quadpy.line_segment.gauss_legendre(m)
                xp = quadrule.points
                xw = quadrule.weights
            elif self.quad_type == 'LGL':
                m = self.p+1
                if n != m:
                    warnings.warn('Mismatch between degree of operator and number of degrees of freedom: n = p+1.')
                quadrule = quadpy.line_segment.gauss_lobatto(m)
                xp = quadrule.points
                xw = quadrule.weights
            elif self.quad_type == 'CSBP':
                raise NotImplementedError
            elif self.quad_type == 'HGT':
                raise NotImplementedError
            elif self.quad_type == 'HGTL':
                raise NotImplementedError
            else:
                raise NotImplementedError
        return xp, xw


# quad = CubatureRules(1, 5, 'LGL')
# xp, xw = quad.quad_line(6)
# print(xp)
# print(xw)