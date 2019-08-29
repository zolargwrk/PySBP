import numpy as np
import quadpy
import orthopy


class Ref1D:
    """ Collects tools for 1D reference element
        Uses the normalized-legendre vandermonde matrix"""

    @staticmethod
    def vandermonde_1d(p, x):
        """ Calculates the vandermonde matrix in 1D"""
        v = np.polynomial.legendre.legvander(x, p)
        for i in range(0, p+1):
            v[:, i] /= np.sqrt(2/(2*i+1))

        return v

    @staticmethod
    def grad_vandermonde_1d(p, x):
        """Calculates the gradient of the vandermonde matrix in 1D"""
        vx = np.zeros((len(x), p+1))
        for i in range(1, p+1):
            jacobi_polynomial = orthopy.line_segment.tree_jacobi(x, i-1, 1, 1, 'normal', symbolic=False)
            jacobi_polynomial = np.asarray(jacobi_polynomial).T
            vx[:, i] = np.sqrt(i*(i+1))*np.asarray(jacobi_polynomial)[:, i-1]
        return vx

    @staticmethod
    def derivative_1d(p, x_ref):
        """Returnes the derivative operator in 1D"""
        v = Ref1D.vandermonde_1d(p, x_ref)
        vx = Ref1D.grad_vandermonde_1d(p, x_ref)
        d_mat = vx @ np.linalg.inv(v)
        return d_mat

    @staticmethod
    def e_mat_1d(tl, tr):
        e_mat = tr @ tr.T - tl @ tl.T
        return e_mat

    @staticmethod
    def projectors_1d(xl_elem, xr_elem, x, **kwargs):
        """Construct the boundary projection matrices
        Inputs: p   - degree of operator
                xl_elem  - left end point of the element
                xr_elem  - right end point of the element
                x   - 1D mesh
                kwargs: scheme = 'LG'  - Legendre-Gauss
                        scheme = 'LGR' - Legendre-Gauss-Radau
                        leave blank for other schemes
        Output: tl - left projection matrix
                tr - right projection matrix"""

        m = len(x)
        tl = np.zeros((m, 1))
        tr = np.zeros((m, 1))

        if ('LG' in list(kwargs.values())) or ('LGR' in list(kwargs.values())):
            for i in range(0, m):
                tl[i] = Ref1D.lagrange(i, xl_elem, x)
                tr[i] = Ref1D.lagrange(i, xr_elem, x)
        else:
            tl[0] = 1
            tr[m-1] = 1

        return tl, tr

    @staticmethod
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

    @staticmethod
    def lift_1d(v, tl, tr):
        e_mat_dg = np.column_stack((tl, tr))
        lift = v @ (v.T @ e_mat_dg)
        return lift




# shp = Ref1D.vandermonde_1d(8, quadpy.line_segment.gauss_lobatto(9).points)
# shpx = Ref1D.grad_vandermonde_1d(8, quadpy.line_segment.gauss_lobatto(9).points)
x = quadpy.line_segment.gauss_lobatto(9).points
d_mat = Ref1D.derivative_1d(8, x)
tl, tr = Ref1D.projectors_1d(-1, 1, x, scheme='LGL')
v = Ref1D.vandermonde_1d(8, x)
lift = Ref1D.lift_1d(v, tl, tr)
# e_mat = Ref1D.e_mat_1d(tl, tr)
print(d_mat)