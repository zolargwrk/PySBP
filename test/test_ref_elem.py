import unittest
import numpy as np
import quadpy
from src.ref_elem import Ref2D_SBP, Ref2D_DG
from types import SimpleNamespace


class TestRef2D_SBP(unittest.TestCase):

    def test_make_sbp_operators2D(self):
        tol = 1e-10
        p = 1
        sbp_family = "omega"
        oper_data = Ref2D_SBP.make_sbp_operators2D(p, sbp_family)
        oper = SimpleNamespace(**oper_data)

        # Vandermonde matrix at cubature nodes
        V = Ref2D_DG.vandermonde_2d(p, oper.r, oper.s)

        # gradient of Vandermond matrix at cubature nodes
        Vder_data = Ref2D_DG.grad_vandermonde2d(p, oper.r, oper.s)
        Vder = SimpleNamespace(**Vder_data)

        # facet Vandermonde matrix
        Vf = oper.Vf

        # Test derivative operator: Dr, Ds
        errDr = np.max(np.abs(oper.Dr @ V - Vder.vdr))
        errDs = np.max(np.abs(oper.Ds @ V - Vder.vds))

        # Test the surface integral operators: Es, Er
        errEr = np.max(np.abs(oper.Qr + oper.Qr.T - oper.Er))
        errEs = np.max(np.abs(oper.Qs + oper.Qs.T - oper.Es))

        # Test the interpolation/extrapolation matrix: R1, R2, R3
        errR1 = np.max(np.abs(oper.R1 @ V - Vf[0]))
        errR2 = np.max(np.abs(oper.R2 @ V - Vf[1]))
        errR3 = np.max(np.abs(oper.R3 @ V - Vf[2]))

        # self.assertEqual(True, False)
        self.assertLessEqual(errDr, tol)
        self.assertLessEqual(errDs, tol)
        self.assertLessEqual(errEr, tol)
        self.assertLessEqual(errEs, tol)
        self.assertLessEqual(errR1, tol)
        self.assertLessEqual(errR2, tol)
        self.assertLessEqual(errR3, tol)


if __name__ == '__main__':
    unittest.main()
