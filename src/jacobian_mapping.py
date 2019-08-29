import numpy as np

class JacobianMapping:
    """Calculates the jacobian"""

    @staticmethod
    def jacobian_1d(x, d_mat_ref):
        """Computes the 1D mesh Jacobian
        inputs: x - nodal location of an element on the physical domain
                d_mat_ref - derivative operator on the reference element
        outputs: rx - the derivative of the reference x with respect to the physical x
                 jac - the transformation Jacobian"""
        jac = d_mat_ref @ x
        rx = 1/jac
        return rx, jac

