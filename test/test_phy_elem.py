import unittest
import numpy as np
from scipy import sparse
from src.assembler import Assembler
from src.rhs_calculator import RHSCalculator, MeshGenerator2D
from src.sats import SATs
from types import SimpleNamespace


class TestPhy2D_SBP(unittest.TestCase):

    def test_poisson_sbp_2d(self):
        tol = 1e-9
        p = 1
        sbp_family = 'gamma'
        flux_type = 'BR2'
        h = 0.5
        nrefine = 1
        nface = 3
        nfp = p+1

        # the rectangular domain
        bL = 0
        bR = 1
        bB = 0
        bT = 1

        # generate mesh
        mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
        btype = ['d', 'd', 'd', 'd']
        ass_data = Assembler.assembler_sbp_2d(p, mesh, btype, sbp_family)
        adata = SimpleNamespace(**ass_data)
        x = adata.x
        y = adata.y
        nelem = adata.nelem
        nnodes = adata.nnodes
        u = 0 * adata.x

        # boundary conditions on a rectangular domain
        uDL_fun = lambda x, y: 0
        uNL_fun = lambda x, y: 0
        uDR_fun = lambda x, y: 0
        uNR_fun = lambda x, y: 0
        uDB_fun = lambda x, y: 0
        uNB_fun = lambda x, y: 0
        uDT_fun = lambda x, y: 0  # np.sin(np.pi * x)
        uNT_fun = lambda x, y: 0

        rhs_data = RHSCalculator.rhs_poisson_sbp_2d(p, u, adata.x, adata.y, adata.r, adata.s, adata.xf, adata.yf, adata.Dr,
                                             adata.Ds, adata.H, adata.B1,adata.B2, adata.B3, adata.R1, adata.R2, adata.R3,
                                             adata.nx, adata.ny, adata.rx, adata.ry, adata.sx, adata.sy,
                                             adata.etoe, adata.etof, adata.bgrp, adata.bgrpD, adata.bgrpN, adata.nelem,
                                             adata.surf_jac, adata.jac, flux_type, uDL_fun, uNL_fun, uDR_fun, uNR_fun,
                                             uDB_fun, uNB_fun, uDT_fun, uNT_fun, bL, bR, bB, bT, None, adata.fscale)
        rdata = SimpleNamespace(**rhs_data)

        # ---test if D2 is correct for every element
        der2_sbp = (rdata.D2B @ ((x.flatten(order='F'))**1 * (y.flatten(order='F'))**(p-1)))
        if p >= 3:
            der2_exact = x.flatten(order='F')**1 * (p-1)*(p-2)*y.flatten(order='F')**(p-3)
        else:
            der2_exact = 0*y.flatten(order='F')

        errD2 = np.linalg.norm(der2_exact - der2_sbp)

        # get operators on the physical elements
        sat_data = SATs.diffusion_sbp_sat_2d_steady(nnodes, nelem, rdata.LxxB, rdata.LxyB, rdata.LyxB, rdata.LyyB,
                                                    adata.Ds, adata.Dr, adata.H, adata.B1, adata.B2, adata.B3,
                                                    adata.R1, adata.R2, adata.R3, adata.rx, adata.ry, adata.sx,
                                                    adata.sy, adata.jac, adata.surf_jac,  adata.nx, adata.ny,
                                                    adata.etoe, adata.etof, adata.bgrp, adata.bgrpD, adata.bgrpN,
                                                    flux_type, rdata.uD, rdata.uN)
        sdata = SimpleNamespace(**sat_data)

        # ---test norm matrix on physical element (the sum of all should give the area)
        area_exact = (bR-bL)*(bT-bB)
        area_sbp = np.sum(sdata.Hg)
        errH = np.linalg.norm(area_exact - area_sbp)

        # ---test the surface norm matrix (its sum should be equal to the length of the facets)
        # get the length of each facet of the physical elements
        lenf1 = np.array([np.sqrt((adata.vx[adata.etov[:, 0]] - adata.vx[adata.etov[:, 1]])**2
                         + (adata.vy[adata.etov[:, 0]] - adata.vy[adata.etov[:, 1]])**2)]).reshape((-1, 1), order="F")
        lenf2 = np.array([np.sqrt((adata.vx[adata.etov[:, 1]] - adata.vx[adata.etov[:, 2]])**2
                         + (adata.vy[adata.etov[:, 1]] - adata.vy[adata.etov[:, 2]])**2)]).reshape((-1, 1), order="F")
        lenf3 = np.array([np.sqrt((adata.vx[adata.etov[:, 2]] - adata.vx[adata.etov[:, 0]])**2
                         + (adata.vy[adata.etov[:, 2]] - adata.vy[adata.etov[:, 0]])**2)]).reshape((-1, 1), order="F")
        lenf_exact = [lenf1, lenf2, lenf3]

        # get the sum of the diagonals of the B matrix at each facet (should be equal to the length of the facet)
        lenf1_sbp = np.sum(np.sum(sdata.BB[0], axis=1), axis=1).reshape(-1, 1)
        lenf2_sbp = np.sum(np.sum(sdata.BB[1], axis=1), axis=1).reshape(-1, 1)
        lenf3_sbp = np.sum(np.sum(sdata.BB[2], axis=1), axis=1).reshape(-1, 1)
        lenf_sbp = [lenf1_sbp, lenf2_sbp, lenf3_sbp]

        # calculate the error
        errB = 0
        for i in range(3):
            errB += np.linalg.norm(lenf_exact[i]-lenf_sbp[i])

        # ---test the derivative operators at each element
        derX_sbp = sparse.block_diag(sdata.DxB) @ ((x.flatten(order='F'))**p)
        derY_sbp = sparse.block_diag(sdata.DyB) @ ((y.flatten(order='F'))**p)
        derX_exact = p*(x.flatten(order='F')**(p-1))
        derY_exact = p*(y.flatten(order='F')**(p-1))

        errDx = np.linalg.norm(derX_exact - derX_sbp)
        errDy = np.linalg.norm(derY_exact - derY_sbp)

        # ---test derivative operator at each facet
        # get facet node locations
        xf1 = (adata.R1 @ x).reshape((-1, 1), order='F')
        xf2 = (adata.R2 @ x).reshape((-1, 1), order='F')
        xf3 = (adata.R3 @ x).reshape((-1, 1), order='F')
        yf1 = (adata.R1 @ y).reshape((-1, 1), order='F')
        yf2 = (adata.R2 @ y).reshape((-1, 1), order='F')
        yf3 = (adata.R3 @ y).reshape((-1, 1), order='F')

        xx = x.reshape((-1, 1), order='F')
        yy = y.reshape((-1, 1), order='F')

        nx1 =(sdata.nxB[0]).reshape((-1, 1))
        ny1 =(sdata.nyB[0]).reshape((-1, 1))
        nx2 =(sdata.nxB[1]).reshape((-1, 1))
        ny2 =(sdata.nyB[1]).reshape((-1, 1))
        nx3 =(sdata.nxB[2]).reshape((-1, 1))
        ny3 =(sdata.nyB[2]).reshape((-1, 1))

        Dgk1_errX = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (xx**p) - p*nx1*xf1**(p-1))
        Dgk2_errX = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (xx**p) - p*nx2*xf2**(p-1))
        Dgk3_errX = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (xx**p) - p*nx3*xf3**(p-1))

        Dgk1_errY = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (yy**p) - p*ny1*yf1**(p-1))
        Dgk2_errY = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (yy**p) - p*ny2*yf2**(p-1))
        Dgk3_errY = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (yy**p) - p*ny3*yf3**(p-1))

        if p>=2:
            Dgk1_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (yy**1 * xx**(p-1))
                                        - ((p-1)*nx1*xf1**(p-2)*yf1 + ny1*yf1**0 * xf1**(p-1)))
            Dgk2_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (yy**1 * xx**(p-1))
                                        - ((p-1)*nx2*xf2**(p-2)*yf2 + ny2*yf2**0 * xf2**(p-1)))
            Dgk3_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (yy**1 * xx**(p-1))
                                        - ((p-1)*nx3*xf3**(p-2)*yf3 + ny3*yf3**0 * xf3**(p-1)))
        else:
            Dgk1_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[0]) @ (yy**1 * xx**(p-1))
                                        - (0*yf1 + ny1*yf1**0 * xf1**(p-1)))
            Dgk2_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[1]) @ (yy**1 * xx**(p-1))
                                        - (0*yf2 + ny2*yf2**0 * xf2**(p-1)))
            Dgk3_errXY = np.linalg.norm(sparse.block_diag(sdata.Dgk[2]) @ (yy**1 * xx**(p-1))
                                        - (0*yf3 + ny3*yf3**0 * xf3**(p-1)))

        errDgk = (Dgk1_errX + Dgk2_errX + Dgk3_errX + Dgk1_errY + Dgk2_errY + Dgk3_errY
                  + Dgk1_errXY + Dgk2_errXY + Dgk3_errXY)

        # check if conditions are met
        self.assertLessEqual(errD2, tol)
        self.assertLessEqual(errH, tol)
        self.assertLessEqual(errB, tol)
        self.assertLessEqual(errDx, tol)
        self.assertLessEqual(errDy, tol)
        self.assertLessEqual(errDgk, tol)



if __name__ == '__main__':
    unittest.main()
