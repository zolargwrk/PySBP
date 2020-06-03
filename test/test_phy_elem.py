import unittest
import numpy as np
from scipy import sparse
from src.assembler import Assembler
from src.rhs_calculator import RHSCalculator, MeshGenerator2D
from src.sats import SATs
from types import SimpleNamespace
from src.calc_tools import CalcTools
from mesh.mesh_tools import MeshTools2D


class TestPhy2D_SBP(unittest.TestCase):

    def test_poisson_sbp_2d(self):
        tol = 1e-7
        p = 4
        sbp_family = 'omega'
        flux_type = 'BR2'
        CurveMesh = True
        p_map = 4
        h = 4

        # the rectangular domain
        bL = 0
        bR = 20
        bB = -5
        bT = 5

        # generate mesh
        mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
        btype = ['d', 'd', 'd', 'd']
        # mesh curvature is applied in the assembler if curved mesh is generated
        ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map=p_map, curve_mesh=CurveMesh)
        adata = SimpleNamespace(**ass_data)
        x = adata.x
        y = adata.y
        nelem = adata.nelem
        nnodes = adata.nnodes
        u = 0 * adata.x

        # boundary conditions on a rectangular domain
        n = m = 1/5
        uDL_fun = lambda x, y: np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
        uNL_fun = lambda x, y: 0
        uDR_fun = lambda x, y: np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
        uNR_fun = lambda x, y: 0
        uDB_fun = lambda x, y: np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
        uNB_fun = lambda x, y: 0
        uDT_fun = lambda x, y: np.sin(m * np.pi * x) * np.sin(n * np.pi * y)  # np.sin(np.pi * x)
        uNT_fun = lambda x, y: 0

        # get operators on physical elements
        phy_data = MeshTools2D.map_operators_to_phy_2d(p, nelem, adata.H, adata.Dr, adata.Ds, adata.Er, adata.Es,
                                                       adata.R1, adata.R2, adata.R3, adata.B1, adata.B2, adata.B3,
                                                       adata.rx, adata.ry, adata.sx, adata.sy, adata.jac,
                                                       adata.surf_jac, adata.nx, adata.ny)
        phy = SimpleNamespace(**phy_data)

        rhs_data = RHSCalculator.rhs_poisson_sbp_2d(u, adata.xf, adata.yf, phy.DxB, phy.DyB, phy.HB, phy.BB, phy.RB,
                                                    phy.nxB, phy.nyB, phy.rxB, phy.ryB, phy.sxB,
                                                    phy.syB, phy.surf_jacB, phy.jacB, adata.etoe, adata.etof,
                                                    adata.bgrp, adata.bgrpD, adata.bgrpN, flux_type, uDL_fun, uNL_fun,
                                                    uDR_fun, uNR_fun, uDB_fun, uNB_fun, uDT_fun, uNT_fun, bL, bR, bB,
                                                    bT, LB=None)
        rdata = SimpleNamespace(**rhs_data)

        # get operators on the physical elements
        sat_data = SATs.diffusion_sbp_sat_2d_steady(nnodes, nelem, rdata.LxxB, rdata.LxyB, rdata.LyxB, rdata.LyyB,
                                                    phy.DxB, phy.DyB, phy.HB, phy.BB, phy.RB, phy.rxB, phy.ryB,
                                                    phy.sxB, phy.syB, phy.jacB, phy.surf_jacB, phy.nxB, phy.nyB,
                                                    adata.etoe, adata.etof, adata.bgrp, adata.bgrpD, adata.bgrpN,
                                                    flux_type, rdata.uD, rdata.uN)
        sdata = SimpleNamespace(**sat_data)

        # initialize error terms
        errD2 = errB = errDx = errDy = errDgk = errH = errQx = errQy = errEx = errEy = err_metric_identityX \
            = err_metric_identityY = 0
        p_numX = p_numY = p
        p_numLaplacian = p-1

        if p_map == 1:
            # ---test if D2 is correct for every element
            der2_sbp = (rdata.D2B @ ((x.flatten(order='F')) ** 1 * (y.flatten(order='F')) ** (p - 1)))
            if p >= 3:
                der2_exact = x.flatten(order='F') ** 1 * (p - 1) * (p - 2) * y.flatten(order='F') ** (p - 3)
            else:
                der2_exact = 0 * y.flatten(order='F')

            errD2 = np.linalg.norm(der2_exact - der2_sbp)

            # -----------------------------------------------------------------------------------------------------
            # ---test B, the surface norm matrix (its sum should be equal to the length of the facets)
            # get the length of each facet of the physical elements
            lenf1 = np.array([np.sqrt((adata.vx[adata.etov[:, 0]] - adata.vx[adata.etov[:, 1]]) ** 2
                                      + (adata.vy[adata.etov[:, 0]] - adata.vy[adata.etov[:, 1]]) ** 2)]).reshape(
                (-1, 1), order="F")
            lenf2 = np.array([np.sqrt((adata.vx[adata.etov[:, 1]] - adata.vx[adata.etov[:, 2]]) ** 2
                                      + (adata.vy[adata.etov[:, 1]] - adata.vy[adata.etov[:, 2]]) ** 2)]).reshape(
                (-1, 1), order="F")
            lenf3 = np.array([np.sqrt((adata.vx[adata.etov[:, 2]] - adata.vx[adata.etov[:, 0]]) ** 2
                                      + (adata.vy[adata.etov[:, 2]] - adata.vy[adata.etov[:, 0]]) ** 2)]).reshape(
                (-1, 1), order="F")
            lenf_exact = [lenf1, lenf2, lenf3]

            # get the sum of the diagonals of the B matrix at each facet (should be equal to the length of the facet)
            lenf1_sbp = np.sum(np.sum(phy.BB[0], axis=1), axis=1).reshape(-1, 1)
            lenf2_sbp = np.sum(np.sum(phy.BB[1], axis=1), axis=1).reshape(-1, 1)
            lenf3_sbp = np.sum(np.sum(phy.BB[2], axis=1), axis=1).reshape(-1, 1)
            lenf_sbp = [lenf1_sbp, lenf2_sbp, lenf3_sbp]

            # calculate the error
            for i in range(3):
                errB += np.linalg.norm(lenf_exact[i] - lenf_sbp[i])

            # -----------------------------------------------------------------------------------------------------
            # ---test Dx, the derivative operators at each element
            q = p
            derX_sbp = sparse.block_diag(phy.DxB) @ ((x.flatten(order='F')) ** q)
            derY_sbp = sparse.block_diag(phy.DyB) @ ((y.flatten(order='F')) ** q)
            derX_exact = q * (x.flatten(order='F') ** (q - 1))
            derY_exact = q * (y.flatten(order='F') ** (q - 1))

            errDx = np.linalg.norm(derX_exact - derX_sbp)
            errDy = np.linalg.norm(derY_exact - derY_sbp)

            # -----------------------------------------------------------------------------------------------------
            # ---test Dgk, derivative operator at each facet
            # get facet node locations
            xf1 = (adata.R1 @ x).reshape((-1, 1), order='F')
            xf2 = (adata.R2 @ x).reshape((-1, 1), order='F')
            xf3 = (adata.R3 @ x).reshape((-1, 1), order='F')
            yf1 = (adata.R1 @ y).reshape((-1, 1), order='F')
            yf2 = (adata.R2 @ y).reshape((-1, 1), order='F')
            yf3 = (adata.R3 @ y).reshape((-1, 1), order='F')

            xx = x.reshape((-1, 1), order='F')
            yy = y.reshape((-1, 1), order='F')

            nx1 = (phy.nxB[0]).reshape((-1, 1))
            ny1 = (phy.nyB[0]).reshape((-1, 1))
            nx2 = (phy.nxB[1]).reshape((-1, 1))
            ny2 = (phy.nyB[1]).reshape((-1, 1))
            nx3 = (phy.nxB[2]).reshape((-1, 1))
            ny3 = (phy.nyB[2]).reshape((-1, 1))

            Dgk1_errX = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (xx ** p) - p * nx1 * xf1 ** (p - 1))
            Dgk2_errX = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (xx ** p) - p * nx2 * xf2 ** (p - 1))
            Dgk3_errX = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (xx ** p) - p * nx3 * xf3 ** (p - 1))

            Dgk1_errY = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (yy ** p) - p * ny1 * yf1 ** (p - 1))
            Dgk2_errY = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (yy ** p) - p * ny2 * yf2 ** (p - 1))
            Dgk3_errY = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (yy ** p) - p * ny3 * yf3 ** (p - 1))

            if p >= 2:
                Dgk1_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (yy ** 1 * xx ** (p - 1))
                                            - ((p - 1) * nx1 * xf1 ** (p - 2) * yf1 + ny1 * yf1 ** 0 * xf1 ** (p - 1)))
                Dgk2_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (yy ** 1 * xx ** (p - 1))
                                            - ((p - 1) * nx2 * xf2 ** (p - 2) * yf2 + ny2 * yf2 ** 0 * xf2 ** (p - 1)))
                Dgk3_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (yy ** 1 * xx ** (p - 1))
                                            - ((p - 1) * nx3 * xf3 ** (p - 2) * yf3 + ny3 * yf3 ** 0 * xf3 ** (p - 1)))
            else:
                Dgk1_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (yy ** 1 * xx ** (p - 1))
                                            - (0 * yf1 + ny1 * yf1 ** 0 * xf1 ** (p - 1)))
                Dgk2_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (yy ** 1 * xx ** (p - 1))
                                            - (0 * yf2 + ny2 * yf2 ** 0 * xf2 ** (p - 1)))
                Dgk3_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (yy ** 1 * xx ** (p - 1))
                                            - (0 * yf3 + ny3 * yf3 ** 0 * xf3 ** (p - 1)))

            errDgk = np.max([Dgk1_errX, Dgk2_errX, Dgk3_errX, Dgk1_errY, Dgk2_errY, Dgk3_errY,
                             Dgk1_errXY, Dgk2_errXY, Dgk3_errXY])

        else:
            # -----------------------------------------------------------------------------------------------------
            # ---test H, norm matrix on physical element (the sum of all should give the area)
            area_exact = (bR - bL) * (bT - bB)
            area_sbp = np.sum(sdata.Hg)
            errH = np.linalg.norm(area_exact - area_sbp)

            # -----------------------------------------------------------------------------------------------------
            # ------ test Q+Q.T = E, i.e., whether SBP property is satisfied
            # get the E matrix in each direction
            errQx = np.max(np.abs(phy.QxB + phy.QxB.transpose(0, 2, 1) - phy.ExB))
            errQy = np.max(np.abs(phy.QyB + phy.QyB.transpose(0, 2, 1) - phy.EyB))

            # -----------------------------------------------------------------------------------------------------
            # ------ test 1.T Ex 1 = 0, surface integral test
            errEx = np.max(np.abs(np.ones((phy.ExB.shape[0], phy.ExB.shape[1], 1)).transpose(0, 2, 1) @ phy.ExB \
                    @ np.ones((phy.ExB.shape[0], phy.ExB.shape[1], 1))))

            errEy = np.max(np.abs(np.ones((phy.EyB.shape[0], phy.EyB.shape[1], 1)).transpose(0, 2, 1) @ phy.EyB \
                                  @ np.ones((phy.EyB.shape[0], phy.EyB.shape[1], 1))))

            # -----------------------------------------------------------------------------------------------------
            # ---- test if the metric identities are satisfied
            DrB = np.block([adata.Dr] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
            DsB = np.block([adata.Ds] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)

            metric_identityX = DrB @ CalcTools.block_diag_to_block_vec(phy.jacB @ phy.rxB) \
                               + DsB @ CalcTools.block_diag_to_block_vec(phy.jacB @ phy.sxB) \

            metric_identityY = DrB @ CalcTools.block_diag_to_block_vec(phy.jacB @ phy.ryB) \
                               + DsB @ CalcTools.block_diag_to_block_vec(phy.jacB @ phy.syB)

            err_metric_identityX = np.max(np.abs(metric_identityX))
            err_metric_identityY = np.max(np.abs(metric_identityY))

        # -----------------------------------------------------------------------------------------------------
        # check accuracy of derivative operators using the exact solution
        errsDx = []
        errsDy = []
        errsD2 = []
        nrefine = 3
        for refine in range(0, nrefine):
            mesh = MeshTools2D.hrefine_uniform_2d(ass_data, bL, bR, bB, bT)

            # update assembled data for 2D implementation
            ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family, p_map, curve_mesh=CurveMesh)
            adata2 = SimpleNamespace(**ass_data)

            x = adata2.x
            y = adata2.y
            nelem = adata2.nelem

            # get operators on physical elements
            phy_data = MeshTools2D.map_operators_to_phy_2d(p, nelem, adata2.H, adata2.Dr, adata2.Ds, adata2.Er, adata2.Es,
                                                           adata2.R1, adata2.R2, adata2.R3, adata2.B1, adata2.B2, adata2.B3,
                                                           adata2.rx, adata2.ry, adata2.sx, adata2.sy, adata2.jac,
                                                           adata2.surf_jac, adata2.nx, adata2.ny)
            phy = SimpleNamespace(**phy_data)

            # get exact solution, derivative, and Laplacian
            ue = lambda x, y: np.sin(m * np.pi * x) * np.sin(n * np.pi * y)     # exact solution
            duedx = lambda x, y: m*np.pi*np.sin(n*np.pi*y)*np.cos(m*np.pi*x)
            duedy = lambda x, y: n*np.pi*np.sin(m*np.pi*x)*np.cos(n*np.pi*y)
            laplacian = lambda x, y: -m**2*np.pi**2*np.sin(np.pi*m*x)*np.sin(np.pi*n*y) \
                                     - n**2*np.pi**2*np.sin(np.pi*m*x)*np.sin(np.pi*n*y)

            # calculate error in derivative
            errDxcurv = (sparse.block_diag(phy.DxB) @ (ue(x, y).T.reshape(-1, 1))).reshape(nelem, -1).T - duedx(x, y)
            errDycurv = (sparse.block_diag(phy.DyB) @ (ue(x, y).T.reshape(-1, 1))).reshape(nelem, -1).T - duedy(x, y)
            errD2curv = (phy.D2B @ (ue(x, y).T.reshape(-1, 1))).reshape(nelem, -1).T - laplacian(x, y)

            # get the maximum error
            errsDx.append(np.max(np.abs(errDxcurv)))
            errsDy.append(np.max(np.abs(errDycurv)))
            errsD2.append(np.max(np.abs(errD2curv)))

        # calculate the trancation error and obtain the order. we devide by ln2 because we are reducing the length
        # of the edges of the triangle by 1/2? and (h1/h2)^p = err1/err2 --> (h1/(h1/2))^p = 2^p = (err1/err2)
        # taking the logarithm of both sides: p = 1/ln2 * ln(err1/err2)
        p_numX = np.min([1/np.log(2) * np.log(errsDx[0]/errsDx[1]), 1/np.log(2) * np.log(errsDx[1]/errsDx[2])])
        p_numY = np.min([1/np.log(2) * np.log(errsDy[0]/errsDy[1]), 1/np.log(2) * np.log(errsDy[1]/errsDy[2])])
        p_numLaplacian = np.min([1/np.log(2) * np.log(errsD2[0]/errsD2[1]), 1/np.log(2) * np.log(errsD2[1]/errsD2[2])])

        # check if conditions are met
        self.assertGreaterEqual(np.min(adata.jac), 1e-06)
        self.assertLessEqual(err_metric_identityX, tol)
        self.assertLessEqual(err_metric_identityY, tol)
        self.assertLessEqual(errD2, tol)
        self.assertLessEqual(errDx, tol)
        self.assertLessEqual(errDy, tol)
        self.assertLessEqual(errQx, tol)
        self.assertLessEqual(errQy, tol)
        self.assertLessEqual(errEx, tol)
        self.assertLessEqual(errEy, tol)
        self.assertLessEqual(errDgk, tol)
        self.assertLessEqual(errH, tol)
        self.assertLessEqual(errB, tol)
        self.assertLessEqual(np.abs(p_numX - p), 1)
        self.assertLessEqual(np.abs(p_numY - p), 1)
        self.assertLessEqual(np.abs(p_numLaplacian - (p-1)), 2)


if __name__ == '__main__':
    unittest.main()
