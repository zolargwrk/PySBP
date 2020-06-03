import unittest
import numpy as np
from src.ref_elem import Ref2D_SBP, Ref2D_DG
from types import SimpleNamespace


class TestRef2D_SBP(unittest.TestCase):

    def test_make_sbp_operators2D(self):
        tol = 1e-12
        p = 3
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
        # Vf = []
        # Vf.append(Ref2D_DG.vandermonde_2d(p, oper.rsf[0][:, 0], oper.rsf[0][:, 1]))
        # Vf.append(Ref2D_DG.vandermonde_2d(p, oper.rsf[1][:, 0], oper.rsf[1][:, 1]))
        # Vf.append(Ref2D_DG.vandermonde_2d(p, oper.rsf[2][:, 0], oper.rsf[2][:, 1]))

        # test derivative operator: Dr, Ds
        errDr = np.max(np.abs(oper.Dr @ V - Vder.vdr))
        errDs = np.max(np.abs(oper.Ds @ V - Vder.vds))

        # test the surface integral operators: Es, Er
        errEr = np.max(np.abs(oper.Qr + oper.Qr.T - oper.Er))
        errEs = np.max(np.abs(oper.Qs + oper.Qs.T - oper.Es))

        # test the interpolation/extrapolation matrix: R1, R2, R3
        # errR1 = np.max(np.abs(oper.R1 @ V - Vf[0]))
        # errR2 = np.max(np.abs(oper.R2 @ V - Vf[1]))
        # errR3 = np.max(np.abs(oper.R3 @ V - Vf[2]))
        errR1 = np.max(np.abs(oper.R1 @ oper.r.flatten()**p - oper.rsf[0][:, 0]**p))
        errR2 = np.max(np.abs(oper.R2 @ oper.r.flatten()**p - oper.rsf[1][:, 0]**p))
        errR3 = np.max(np.abs(oper.R3 @ oper.r.flatten()**p - oper.rsf[2][:, 0]**p))

        # test accuracy of surface integral: E
        # to obtain the analytical surface integral use the fact that: int(v*du) + int(u*dv) = int_surface(uv)
        # setting v = 1 gives int(du) = int_surface(uv)
        q = 2*p
        u = oper.s
        surf_integral = ((u**p).T @ (oper.Er + oper.Es) @ (u ** p))[0]
        analytical_surf_integral = -q/(q+1) + 1 + (-1)**(q+1)*q/(q+1) - (-1)**q
        errE = np.abs(surf_integral - analytical_surf_integral)

        # test the line integral matrix on each facet: B1, B2, B3
        t = 2*p
        analytical_line_B1 = - 1/(t+1) * ((-1)**(t+1) - 1)          # line integral of r**t on facet 1
        line_B1 = np.diag(oper.B1).T @ oper.rsf[0][:, 0]**t
        errB1 = np.abs(line_B1 - analytical_line_B1)

        analytical_line_B2 = np.sqrt(2)/(t+1) *(1 - (-1)**(t+1))     # line integral of r**t on facet 2
        line_B2 = np.diag(oper.B2).T @ oper.rsf[1][:, 0]**t
        errB2 = np.abs(line_B2 - analytical_line_B2)

        analytical_line_B3 = 1/(t+1) *(1 - (-1)**(t+1))             # line integral of s**t on facet 3
        line_B3 = np.diag(oper.B3).T @ oper.rsf[2][:, 1] ** t
        errB3 = np.abs(line_B3 - analytical_line_B3)

        # tests the norm matrix: H
        H_test = lambda pL, pR: ((u**pL).T @ oper.H @ (pR * u**(pR-1))).flatten()[0]
        errH = np.abs(H_test(0, q) - analytical_surf_integral)

        H_test2 = ((u ** p).T @ oper.H @ (oper.Ds @ (u ** (p)))).flatten()[0]
        errH2 = np.abs(H_test2 - H_test(p, p))

        # test compatibility: v^T@H@du + u^T@ H @v = u^T @ E @ v
        errComp = np.abs(H_test(p, p) + H_test(p, p) - surf_integral)[0]

        # check the decomposition of the E matrix
        Er_decomp = oper.nx[0] * oper.R1.T @ oper.B1 @ oper.R1 + oper.nx[1] * oper.R2.T @ oper.B2 @ oper.R2 \
                    + oper.nx[2] * oper.R3.T @ oper.B3 @ oper.R3
        Es_decomp = oper.ny[0] * oper.R1.T @ oper.B1 @ oper.R1 + oper.ny[1] * oper.R2.T @ oper.B2 @ oper.R2 \
                    + oper.ny[2] * oper.R3.T @ oper.B3 @ oper.R3

        Er_decomp_err = np.max(np.abs(Er_decomp - oper.Er)) - tol
        Es_decomp_err = np.max(np.abs(Es_decomp - oper.Es)) - tol

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
        self.assertLessEqual(errH, tol)
        self.assertLessEqual(errH2, tol)
        self.assertLessEqual(errE, tol)
        self.assertLessEqual(errComp, tol)
        self.assertLessEqual(Er_decomp_err, tol)
        self.assertLessEqual(Es_decomp_err, tol)


if __name__ == '__main__':
    unittest.main()
