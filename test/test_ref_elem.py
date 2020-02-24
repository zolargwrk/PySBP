import unittest
import numpy as np
import quadpy
from src.ref_elem import Ref2D_SBP, Ref2D_DG
from types import SimpleNamespace


class TestRef2D_SBP(unittest.TestCase):

    def test_make_sbp_operators2D(self):
        tol = 1e-10
        p = 4
        sbp_family = "diage"
        oper_data = Ref2D_SBP.make_sbp_operators2D(p, sbp_family)
        oper = SimpleNamespace(**oper_data)

        # Vandermonde matrix at cubature nodes
        V = Ref2D_DG.vandermonde_2d(p, oper.r, oper.s)

        # gradient of Vandermond matrix at cubature nodes
        Vder_data = Ref2D_DG.grad_vandermonde2d(p, oper.r, oper.s)
        Vder = SimpleNamespace(**Vder_data)

        # facet Vandermonde matrix
        Vf = oper.Vf

        # test derivative operator: Dr, Ds
        errDr = np.max(np.abs(oper.Dr @ V - Vder.vdr))
        errDs = np.max(np.abs(oper.Ds @ V - Vder.vds))

        # test the surface integral operators: Es, Er
        errEr = np.max(np.abs(oper.Qr + oper.Qr.T - oper.Er))
        errEs = np.max(np.abs(oper.Qs + oper.Qs.T - oper.Es))

        # test the interpolation/extrapolation matrix: R1, R2, R3
        errR1 = np.max(np.abs(oper.R1 @ V - Vf[0]))
        errR2 = np.max(np.abs(oper.R2 @ V - Vf[1]))
        errR3 = np.max(np.abs(oper.R3 @ V - Vf[2]))
        # errR1 = np.max(np.abs(oper.R1 @ oper.r.flatten()**p - oper.rsf[0][:, 0]**p))
        # errR2 = np.max(np.abs(oper.R2 @ oper.r.flatten()**p - oper.rsf[1][:, 0]**p))
        # errR3 = np.max(np.abs(oper.R3 @ oper.r.flatten()**p - oper.rsf[2][:, 0]**p))

        # test accuracy of surface integral: E
        # to obtain the analytical surface integral use the fact that: int(v*du) + int(u*dv) = int_surface(uv)
        # setting v = 1 gives int(du) = int_surface(uv)
        q = 2*p
        u = oper.r
        surf_integral = ((u**0).T @ (oper.Er + oper.Es) @ (u ** q))[0]
        analytical_surf_integral = -q/(q+1) + 1 + (-1)**(q+1)*q/(q+1) - (-1)**q
        errE = np.abs(surf_integral - analytical_surf_integral)

        # test the line integral matrix on each facet: B1, B2, B3
        t = 2*p
        analytical_line_B1 = - np.sqrt(2)/(t+1) * ((-1)**(t+1) - 1)     # line integral of r**t on facet 1
        line_B1 = np.diag(oper.B1).T @ oper.rsf[0][:, 0]**t
        errB1 = np.abs(line_B1 - analytical_line_B1)

        analytical_line_B2 = 1/(t+1) *(1 - (-1)**(t+1))                 # line integral of s**t on facet 2
        line_B2 = np.diag(oper.B2).T @ oper.rsf[1][:, 1]**t
        errB2 = np.abs(line_B2 - analytical_line_B2)

        analytical_line_B3 = 1/(t+1) *(1 - (-1)**(t+1))                 # line integral of r**t on facet 3
        line_B3 = np.diag(oper.B3).T @ oper.rsf[2][:, 0] ** t
        errB3 = np.abs(line_B3 - analytical_line_B3)

        # self.assertEqual(True, False)
        self.assertLessEqual(errDr, tol)
        self.assertLessEqual(errDs, tol)
        self.assertLessEqual(errEr, tol)
        self.assertLessEqual(errEs, tol)
        self.assertLessEqual(errR1, tol)
        self.assertLessEqual(errR2, tol)
        self.assertLessEqual(errR3, tol)
        self.assertLessEqual(errB1, tol)
        self.assertLessEqual(errB2, tol)
        self.assertLessEqual(errB3, tol)
        self.assertLessEqual(errE, tol)


if __name__ == '__main__':
    unittest.main()
