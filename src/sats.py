import numpy as np
from collections import deque
from scipy import sparse
import scipy as sp
from src.calc_tools import CalcTools
import matplotlib.pyplot as plt

class SATs:

    @staticmethod
    def diffusion_dg_sat_1d(var, var_type, tl, tr, vmapM, vmapP, nx, mapI, mapO, varin, varout, rx, flux_type='BR1',
                            du=None, ux=None, uxin=None, uxout=None):
        # set var_type to 'u' or 'q' depending on which variable you want to apply penalty SATs
        # for BR2 you need to insert du
        nface = 2
        n = var.shape[0]
        nelem = var.shape[1]

        # project var to the interfaces to get the var-var* at the faces
        var_proj = var.copy()
        var_projM = (var.T @ tl)[:, 0]
        var_projP = (var.T @ tr)[:, 0]
        var_proj[0, :] = var_projM
        var_proj[n-1, :] = var_projP

        dvar = np.zeros((nface, nelem))
        dvar = dvar.reshape((nface * nelem, 1), order='F')

        # reshape some matrices and calculate dvar
        var_projF = var_proj.reshape((n * nelem, 1), order='F')
        nx_tempC = nx.reshape((nface * nelem, 1), order='C')
        var_proj = np.reshape(np.array([var_projM, var_projP]), (2*len(var_projM), 1), order='F')

        if flux_type == 'BR1':
            dvar[0:] = (var_projF[vmapM][:, 0] - var_projF[vmapP][:, 0]) / 2
            # set boundary sats
            dvar[mapI] = (var_proj[mapI] - varin) / 2
            dvar[mapO] = (var_proj[mapO] - varout) / 2
        elif flux_type == 'BRZ':
            if var_type=='u':
                dvar[0:] = (var_projF[vmapM][:, 0] - var_projF[vmapP][:, 0]) / 2
                # set boundary sats
                dvar[mapI] = (var_proj[mapI] - varin) / 2
                dvar[mapO] = (var_proj[mapO] - varout) / 2
            elif var_type=='q':
                dvar[0:] = (var_projF[vmapM][:, 0] - var_projF[vmapP][:, 0])/2
                # set boundary sats
                dvar[mapI] = var_proj[mapI] - varin
                dvar[mapO] = var_proj[mapO] - varout

                alpha_r = 1
                dvar = dvar + alpha_r*nx_tempC*(du.reshape((nface * nelem, 1), order='F'))

        elif flux_type == 'IP':

            if var_type == 'u':
                dvar[0:] = (var_projF[vmapM][:, 0] - var_projF[vmapP][:, 0])/2
                # set boundary sats
                dvar[mapI] = (var_proj[mapI] - varin)/2
                dvar[mapO] = (var_proj[mapO] - varout)/2
            elif var_type == 'q':
                ux_proj = ux.copy()
                ux_projM = (ux.T @ tl)[:, 0]
                ux_projP = (ux.T @ tr)[:, 0]
                ux_proj[0, :] = ux_projM
                ux_proj[n - 1, :] = ux_projP
                ux_projF = ux_proj.reshape((n * nelem, 1), order='F')
                ux_proj = np.reshape(np.array([ux_projM, ux_projP]), (2 * len(ux_projM), 1), order='F')

                dvar[0:] = var_projF[vmapM][:, 0] - (ux_projF[vmapM][:, 0] + ux_projF[vmapP][:, 0]) / 2
                # set boundary sats
                dvar[mapI] = var_proj[mapI] - (ux_proj[mapI] + uxin)/2
                dvar[mapO] = var_proj[mapO] - (ux_proj[mapO] + uxout)/2

                hmin = 2 / (np.max(rx))
                alpha_r = n ** 2 / hmin
                dvar = dvar + alpha_r * nx_tempC * (2*du.reshape((nface * nelem, 1), order='F'))

        dvar = dvar.reshape((nface, nelem), order='F')
        return dvar

    @staticmethod
    def diffusion_sbp_sat_1d(u, d_mat, d2_mat, h_mat, tl, tr, vmapM, vmapP, nx, mapI, mapO, uin, uout, rx,
                             flux_type='BR1', boundary_type='Periodic', db_mat=None, b=1, app=1,
                             uD_left=None, uD_right=None, uN_left=None, uN_right=None):
        nface = 2
        n = np.shape(u)[0]
        nelem = np.shape(u)[1]
        tr = tr.reshape((n, 1))
        tl = tl.reshape((n, 1))

        # scale the matrices (the ones given are for the reference element)
        h_mat = 1/rx[0, 0]*h_mat
        d_mat = rx[0, 0]*d_mat
        db_mat = rx[0, 0]*db_mat
        d2_mat = d2_mat
        h_inv = np.linalg.inv(h_mat)
        b_mat = np.diag(b.flatten())

        if app ==2:
            db_inv = np.linalg.inv(db_mat)
            # A = db_inv.T @ d_mat.T @ h_mat @ d_mat @ db_inv

            # construct the A matrix for CSBP_Mattsson2004 operators
            # e_mat = np.zeros((n,n))
            # e_mat[0, 0] = -1
            # e_mat[n-1,n-1] = 1
            e_mat = tr@tr.T - tl@tl.T
            A = db_inv.T @(e_mat @ b_mat @ db_mat - h_mat @ d2_mat)@ db_inv

            M = np.linalg.pinv(A)

        # SAT coefficients for different methods
        T1 = 0
        T2k = 0
        T2v = 0
        T3k = 0
        T3v = 0
        T4k = 0
        T4v = 0
        T5k = 0
        T5v = 0
        T6k = 0
        T6v = 0

        if flux_type == 'BR1':
            if app == 2:
                eta = 2
                T1 = (eta / 4 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                eta = 1
                T1 = (eta/4*(tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]
            T2k = (-1/2)
            T3k = (1/2)
            T2v = (-1/2)
            T3v = (1/2)
            T5k = (-1/4*(tr.T @ b_mat @ h_inv @ tl))[0, 0]
            T5v = (-1/4*(tl.T @ b_mat @ h_inv @ tr))[0, 0]
            T6k = (1/4*(tr.T @ b_mat @ h_inv @ tr))[0, 0]
            T6v = (1/4*(tl.T @ b_mat @ h_inv @ tl))[0, 0]

        elif flux_type == 'BRZ':
            eta = nface
            if app == 2:
                T1 = (eta/ 4 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                T1 = ((1+eta)/4 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]
            T2k = (-1 / 2)
            T3k = (1 / 2)
            T2v = (-1 / 2)
            T3v = (1 / 2)
            T5k = (-1 / 4 * (tr.T @ b_mat @ h_inv @ tl))[0, 0]
            T5v = (-1 / 4 * (tl.T @ b_mat @ h_inv @ tr))[0, 0]
            T6k = (1 / 4 * (tr.T @ b_mat @ h_inv @ tr))[0, 0]
            T6v = (1 / 4 * (tl.T @ b_mat @ h_inv @ tl))[0, 0]

        elif flux_type == 'BR2':
            eta = 2
            if app == 2:
                T1 = (eta/4 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                T1 = (eta/4 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]

            T2k = (-1/2)
            T3k = (1/2)
            T2v = (-1/2)
            T3v = (1/2)

        elif flux_type == 'BR22':
            eta = 2
            T1 = (eta / 4 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]
            T2k = (-1 / 4)
            T3k = (-1 / 2)
            T2v = (-3 / 4)
            T3v = (3 / 2)

        elif flux_type == 'IP':
            T1 = 1 / 2 * (np.linalg.norm((tr.T @ b_mat @ (h_inv ** (1 / 2))), 1)) ** 2 \
                 + 1 / 2 * (np.linalg.norm((tl.T @ b_mat @ (h_inv ** (1 / 2))), 1)) ** 2
            T2k = (-1 / 2)
            T3k = (1 / 2)
            T2v = (-1 / 2)
            T3v = (1 / 2)
        elif flux_type == 'BO':
            T2k = (1/2)
            T3k = (1/2)
            T2v = (1/2)
            T3v = (1/2)
        elif flux_type == 'NIPG':
            eta = nface
            he = rx[0, 0]
            mu = eta / he
            T1 = mu
            T2k = (1 / 2)
            T3k = (1 / 2)
            T2v = (1 / 2)
            T3v = (1 / 2)
        elif flux_type == 'CNG':
            eta = 2
            if app == 2:
                T1 = (eta / 16 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                T1 = (eta / 16 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]

            T3k = 1/2
            T3v = 1/2
        elif flux_type == 'LDG' or flux_type == 'CDG':
            eta = nface
            if app == 2:
                T1 = (eta * (tr.T @ M @ tr ))[0, 0]
            else:
                T1 = eta*(tr.T @ b_mat @ h_inv @ tr)[0, 0]
            # T1 = eta * (tr.T @ M @ tr )[0, 0]  #-2134 # unstable for p=4 HGT operator
            T2k = -1
            T2v = 0
            T3k = 0
            T3v = 1

        # project solution to interfaces
        u_projk = (tr.T @ u).flatten()    # to the right interface
        u_projv = (tl.T @ u).flatten()   # to the left interface

        # rotate projected solution to match right and left projections at an interface
        u_projP = deque(u_projv)
        u_projP.rotate(-1)
        u_projP = np.asarray(u_projP)
        u_projM = deque(u_projk)
        u_projM.rotate(1)
        u_projM = np.asarray(u_projM)

        # jump in solution
        Uk = u_projk - u_projP
        Uv = u_projv - u_projM

        # calculate normal derivative at the interfaces
        if app==2:
            Dgk = nx[0, 1] * (tr.T @ b_mat @ db_mat)  # D_{\gamma k}
            Dgv = nx[0, 0] * (tl.T @ b_mat @ db_mat)  # D_{\gamma v}
        else:
            Dgk = nx[0, 1]*(tr.T @ b_mat @ d_mat)   # D_{\gamma k}
            Dgv = nx[0, 0]*(tl.T @ b_mat @ d_mat)   # D_{\gamma v}

        # project derivative of solution on to interfaces
        Dnk = (Dgk @ u).flatten()
        Dnv = (Dgv @ u).flatten()

        # rotate to match right and left projections at an interface
        DnP = deque(Dnv)
        DnP.rotate(-1)
        DnP = np.asarray(DnP)
        DnM = deque(Dnk)
        DnM.rotate(1)
        DnM = np.asarray(DnM)

        # extended SAT terms
        Uvv = Uv
        Ukk = Uk
        Ukkk = deque(Uk)
        Ukkk.rotate(-1)
        Ukkk = np.asarray(Ukkk)
        Uvvv = deque(Uv)
        Uvvv.rotate(1)
        Uvvv = np.asarray(Uvvv)

        # jump in derivative
        Dk = (Dnk + DnP)
        Dv = (Dnv + DnM)

        # coefficient matrix C and left and right multipliers matrices in equation for the interface SATs
        Ck = np.array([[T1, T3k], [T2k, T4k]])
        Cv = np.array([[T1, T3v], [T2v, T4v]])
        Vk = np.block([[Uk], [Dk]])
        Vv = np.block([[Uv], [Dv]])
        Wk = np.block([[tr.reshape(n, 1), Dgk.reshape(n, 1)]])
        Wv = np.block([[tl.reshape(n, 1), Dgv.reshape(n, 1)]])

        # SAT terms to the right, sk, and to the left, sv, interfaces of an element
        sat_k = Wk @ Ck @ Vk
        sat_v = Wv @ Cv @ Vv

        # 1st neighbor extended SAT terms
        sat_M = tr @ (T5k * Uvv.reshape(1, nelem))
        sat_P = tl @ (T5v * Ukk.reshape(1, nelem))

        # 2nd neighbour extended SAT terms
        sat_PP = tr @ (T6k * Ukkk.reshape(1, nelem))
        sat_MM = tl @ (T6v * Uvvv.reshape(1, nelem))

        # boundary SATs for non-periodic boundary condition
        if boundary_type != 'Periodic':
            if app == 2:
                TD_left = 2 * (tl.T @ M @ tl)[0, 0]
                TD_right = 2 * (tr.T @ M @ tr)[0, 0]
            else:
                TD_left = 2 * (tl.T @ b_mat @ h_inv @ tl)[0, 0]  # Dirichlet boundary flux coefficient at the left bc
                TD_right = 2 * (tr.T @ b_mat @ h_inv @ tr)[0, 0]  # Dirichlet boundary flux coefficient at the right bc
            CDk = np.array([[TD_right], [-1]])  # coefficient for dirichlet boundary at the right boundary
            CDv = np.array([[TD_left], [-1]])  # coefficient for dirichlet boundary at the left boundary
            DnNk = (Dgk @ u[:, -1]).flatten()  # derivative of u at right boundary (for Neumann bc implementation)
            DnNv = (Dgv @ u[:, 0]).flatten()  # derivative of u at left boundary (for Neumann bc implementation)
            WDk = np.block([[tr.reshape(n, 1), Dgk.reshape(n, 1)]])
            WDv = np.block([[tl.reshape(n, 1), Dgv.reshape(n, 1)]])

            sD_left = 0
            sD_right = 0
            sN_left = 0
            sN_right = 0
            fD_left = 0
            fD_right = 0
            fN_left = 0
            fN_right = 0

            if uD_left != None:
                UDv = u_projv[0]   # difference in u and Dirichlet bc at the left boundary
                sD_left = (WDv @ CDv) * UDv
                fD_left =  -(WDv @ CDv) *uD_left
            if uD_right != None:
                UDk = u_projk[-1]  # difference in u and Dirichlet bc at the right boundary
                sD_right = (WDk @ CDk) * UDk
                fD_right = -(WDk @ CDk)* uD_right
            if uN_left != None:
                sN_left = tl.T * DnNv
                fN_left = -tl.T *uN_left
            if uN_right != None:
                sN_right = tr.T * DnNk
                fN_right = -tr.T *uN_right

            sB_left = sD_left + sN_left
            sB_right = sD_right + sN_right

            fB_left = h_inv@(fD_left + fN_left)
            fB_right = h_inv@(fD_right + fN_right)


            c = 0     # a constant to make the coefficient 1/4 instead of 2 for T5 and T6
            sat_v[:, 0] = sB_left[:, 0]
            sat_k[:, -1] = sB_right[:, 0]

            sat_M[:, 0] = c * sB_left[:, 0]
            sat_P[:, -1] = c * sB_right[:, 0]

            if flux_type == 'BR1' or flux_type == 'BRZ':
                sat_MM[:, 0] = c*sB_left[:, 0]
                sat_PP[:, -1] = c*sB_right[:, 0]
                sat_MM[:, 1] = c*sB_left[:, 0]
                sat_PP[:, -2] = c*sB_right[:, 0]

        sI = h_inv @ (sat_k + sat_v + sat_P + sat_M + sat_PP + sat_MM)
        fB = np.zeros(u.shape)
        fB[:,0] = fB[:,0]+fB_left.flatten()
        fB[:,nelem-1] = fB[:,nelem-1]+fB_right.flatten()
        # sI = h_inv @ (sat_k + sat_v + sat_P + sat_M )


        # # -------- delete  when done testing --------
        # d_inv = np.linalg.pinv(d_mat)
        # Dd = d_mat @ db_inv
        # vv = Dd.T @ h_mat @ Dd
        # vv_inv = np.linalg.pinv((Dd.T @ h_mat @ Dd))
        # aa = vv @ vv_inv
        # kk = np.zeros((db_mat.shape[0], 1))
        # kk[0] = 0
        # kk[-1] = 1
        # I = np.eye((aa.shape[0]))
        # aaa = (I - aa) @ kk
        uu, ss, vh = np.linalg.svd(A)
        vv2 = uu @ np.diag(ss) @ vh
        # Ir = (np.diag(ss) @ np.linalg.pinv(np.diag(ss)))
        # jj = uu @ Ir @ uu.T
        # N = np.ones((30, 1))
        # N2 = np.ones((30, 1))
        # N2[0, 0] = 0
        # N2[-1, -1] = 0
        # f, tt, fh = np.linalg.svd(vv)
        # dhd = d_mat.T @ h_mat @ d_mat
        #
        # IV = np.block([[vv, 0 * vv], [0 * vv, vv]])
        # ZZ = np.linalg.pinv(IV) - np.block([[np.linalg.pinv(vv), 0 * vv], [0 * vv, np.linalg.pinv(vv)]])
        #
        #
        # zz = np.zeros((n,n))
        # T1 = T1
        # T2 = T2k
        # T3 = T3k
        # TD = TD_left
        #
        # N4 = np.block([[T1, -T1, (T3-1)*tr.T, T3*(-tl.T) ], [-T1, T1, T3*tr.T, (T3-1)*(-tl.T)],
        #               [tr*T2, -tr*T2, 1/2*A, zz ], [-(-tl)*T2, (-tl)*T2, zz, 1/2*A]])
        #
        # N2 = np.block([[TD, -tl.T], [-tl, 1/2*A]])
        #
        # eigN4 = np.sort(np.linalg.eigvals(N4)).reshape(N4.shape[0], 1)
        # eigN2 = np.sort(np.linalg.eigvals(N2)).reshape(N2.shape[0], 1)
        #
        uu, ss, vh = np.linalg.svd(db_inv.T @ e_mat-db_inv.T@ h_mat @ d2_mat @ db_inv)
        a = (1 + db_mat[0, 0]) / db_mat[0, 0] - 1
        Acorner = a * (-1 * db_mat[0, 0] - h_mat[0, 0] * d2_mat[0, 0]) * a
        Mcorner = (1 / Acorner)*np.sqrt(nelem)*n - M[0, 0]
        h = 2  / (n-1)
        M0 = M[0, 0]*h - 1267.2456728069847
        q = h*T1
        # # print(q)
        #----------end delete-----------

        return sI, fB

    @staticmethod
    def diffusion_sbp_sat_1d_steady(n, nelem, d_mat, d2_mat, h_mat, tl, tr, nx, rx,
                                    flux_type='BR1', boundary_type='Periodic', db_mat=None, b=1, app=1,
                                    uD_left=None, uD_right=None, uN_left=None, uN_right=None):
        nface = 2
        tr = tr.reshape((n, 1))
        tl = tl.reshape((n, 1))

        # scale the matrices (the ones given are for the reference element)
        h_mat = 1/rx[0, 0]*h_mat
        d_mat = rx[0, 0]*d_mat
        db_mat = rx[0, 0]*db_mat
        d2_mat = d2_mat     # it's already scaled (it's called here after it is multiplied by rx)
        h_inv = np.linalg.inv(h_mat)
        b_mat = np.diag(b.flatten())

        # calculate normal derivative at the interfaces
        if app == 2:
            Dgk = nx[0, 1] * (tr.T @ b_mat @ db_mat)  # D_{\gamma k}
            Dgv = nx[0, 0] * (tl.T @ b_mat @ db_mat)  # D_{\gamma v}
        else:
            Dgk = nx[0, 1] * (tr.T @ b_mat @ d_mat)  # D_{\gamma k}
            Dgv = nx[0, 0] * (tl.T @ b_mat @ d_mat)  # D_{\gamma v}

        if app ==2:
            db_inv = np.linalg.inv(db_mat)
            # A = db_inv.T @ d_mat.T @ h_mat @ d_mat @ db_inv

            # construct the A matrix for CSBP_Mattsson2004 operators
            e_mat = tr @ tr.T - tl @ tl.T

            if b_mat[0,0] !=0:
                A = db_inv.T @ (e_mat @ b_mat @ db_mat - h_mat @ d2_mat) @ db_inv
                M = np.linalg.pinv(A)
                # kk = np.diag(1/rx[:,0].flatten()) @ M # kk is constant for each operator

                # # In Eriksson's paper, we have Erk = M^{-1} = S A^{-1} S and q as
                # Erk = db_mat @ np.linalg.pinv(e_mat @ b_mat @ db_mat - h_mat @ d2_mat) @ db_mat.T
                # q_Erk = 1/((nelem*len(h_mat) -1) - (nelem-1)) * (tl.T @ Erk @ tl)
                # qh = 1/((nelem*len(h_mat) -1) - (nelem-1)) *(tl.T @ M @ tl)
                # print(np.abs(qh-q_Erk))
                # nlsp = sp.linalg.null_space(M)
            else:
                M = 0*db_mat

        # SAT coefficients for different methods
        T1 = 0
        T2k = 0
        T2v = 0
        T3k = 0
        T3v = 0
        T4k = 0
        T4v = 0
        T5k = 0
        T5v = 0
        T6k = 0
        T6v = 0

        if flux_type == 'BR1':
            if app == 2:
                eta = 2
                T1 = (eta / 4 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                eta = 1
                T1 = (eta/4*(tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]
            T2k = (-1/2)
            T3k = (1/2)
            T2v = (-1/2)
            T3v = (1/2)
            T5k = (-1/4*(tr.T @ b_mat @ h_inv @ tl))[0, 0]
            T5v = (-1/4*(tl.T @ b_mat @ h_inv @ tr))[0, 0]
            T6k = (1/4*(tr.T @ b_mat @ h_inv @ tl))[0, 0]
            T6v = (1/4*(tl.T @ b_mat @ h_inv @ tr))[0, 0]

        elif flux_type == 'BRZ':
            eta = nface
            if app == 2:
                T1 = (eta/ 4 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                T1 = ((1+eta)/4 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]
            T2k = (-1 / 2)
            T3k = (1 / 2)
            T2v = (-1 / 2)
            T3v = (1 / 2)
            T5k = (-1 / 4 * (tr.T @ b_mat @ h_inv @ tl))[0, 0]
            T5v = (-1 / 4 * (tl.T @ b_mat @ h_inv @ tr))[0, 0]
            T6k = (1 / 4 * (tr.T @ b_mat @ h_inv @ tl))[0, 0]
            T6v = (1 / 4 * (tl.T @ b_mat @ h_inv @ tr))[0, 0]

        elif flux_type == 'BR2':
            if app == 2:
                eta = 2
                T1 = (eta/4 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                eta = 2
                T1 = (eta/4 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]

            T2k = (-1/2)
            T3k = (1/2)
            T2v = (-1/2)
            T3v = (1/2)

        elif flux_type == 'BR22':
            eta = 2
            T1 = (eta / 4 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]
            T2k = (-1 / 4)
            T3k = (-1 / 2)
            T2v = (-3 / 4)
            T3v = (3 / 2)

        elif flux_type == 'IP':
            T1 = 1 / 2 * (np.linalg.norm((tr.T @ (b_mat @ h_inv ** (1 / 2))), 1)) ** 2 \
                 + 1 / 2 * (np.linalg.norm((tl.T @ (b_mat @ h_inv ** (1 / 2))), 1)) ** 2
            T2k = (-1 / 2)
            T3k = (1 / 2)
            T2v = (-1 / 2)
            T3v = (1 / 2)
        elif flux_type == 'BO':
            T2k = (1/2)
            T3k = (1/2)
            T2v = (1/2)
            T3v = (1/2)
        elif flux_type == 'NIPG':
            eta = nface
            he = rx[0, 0]
            mu = eta / he
            T1 = mu
            T2k = (1 / 2)
            T3k = (1 / 2)
            T2v = (1 / 2)
            T3v = (1 / 2)
        elif flux_type == 'CNG':
            eta = 2
            if app == 2:
                T1 = (eta/ 16 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                T1 = (eta/ 16 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]

            T3k = 1/2
            T3v = 1/2
            T2k = 0
            T2v = 0

        elif flux_type == 'LDG' or flux_type == 'CDG':
            eta = 2
            if app == 2:
                T1 = (eta * (tr.T @ M @ tr))[0, 0]
            else:
                T1 = eta*(tr.T @ b_mat @ h_inv @ tr)[0, 0]
            # T1 = eta * (tr.T @ M @ tr )[0, 0]  #-2134 # unstable for p=4 HGT operator
            T2k = -1
            T2v = 0
            T3k = 0
            T3v = 1

        # boundary SATs for boundary condition
        sD_left = 0
        sD_right = 0
        sN_left = 0
        sN_right = 0

        if uD_left != None:
            sD_left = 1
        if uD_right != None:
            sD_right = 1
        if uN_left != None:
            sN_left = 1
        if uN_right != None:
            sN_right = 1

        if app == 2:
            TD_left = sD_left * 2 * (tl.T @ M @ tl)[0, 0]
            TD_right = sD_right * 2 * (tr.T @ M @ tr)[0, 0]
        else:
            TD_left = sD_left *2 * (tl.T @ b_mat @ h_inv @ tl)[0, 0]  # Dirichlet boundary flux coefficient at the left bc
            TD_right = sD_right * 2 * (tr.T @ b_mat @ h_inv @ tr)[0, 0]  # Dirichlet boundary flux coefficient at the right bc


        sat_p1 = 0
        sat_p2 = 0
        sat_m1 = 0
        sat_m2 = 0
        sat_p2 = 0
        sat_m2 = 0

        # construct diagonals of the SAT matrix
        if nelem >= 2:
            sat_diag = h_inv @ (T1 * (tr @ tr.T) + T3k * (tr @ Dgk) + T2k * (Dgk.T  @ tr.T))\
                       + h_inv @ (T1 * (tl @ tl.T) + T3v * (tl @ Dgv) + T2v * (Dgv.T @ tl.T))\
                       + h_inv @ (T5k * (tr @ tl.T) + T5v * (tl @ tr.T))
            sat_diag_p1 = h_inv @ (-T1 * (tr @ tl.T) + T3k * (tr @ Dgv) - T2k * (Dgk.T @ tl.T)
                                   - T5v * (tl @ tl.T) + T6k * (tr @ tr.T))
            sat_diag_m1 = h_inv @ (-T1 * (tl @ tr.T) + T3v * (tl @ Dgk) - T2v * (Dgv.T @ tr.T)
                          - T5k * (tr @ tr.T) + T6v * (tl @ tl.T))
            sat_diag_p2 = - h_inv @ (T6k * (tr @ tl.T))
            sat_diag_m2 = - h_inv @ (T6v * (tl @ tr.T))

            sat_diags = sat_diag.copy().repeat(nelem).reshape(nelem,n,n, order='F').transpose(0,2,1)
            sat_diags_p1 = sat_diag_p1.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)
            sat_diags_p2 = sat_diag_p2.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)
            sat_diags_m1 = sat_diag_m1.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)
            sat_diags_m2 = sat_diag_m2.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)
            sat_diags_p1[nelem - 1, :, :] = sat_diags_p1[nelem - 1, :, :] - h_inv @ (T6k * (tr @ tr.T))
            sat_diags_m1[0, :, :] = sat_diags_m1[0, :, :] - (T6v * h_inv @ (tl @ tl.T))

            # set boundary conditions
            sat_diags[0,:,:] = h_inv @ (TD_left * (tl @ tl.T)  - sD_left * Dgv.T  @ tl.T +sN_left* tl @ Dgv
                                        + T1 * (tr @ tr.T) + T5k * (tl @ tr.T)
                                        + T3k * (tr @ Dgk) + T2k * (Dgk.T  @ tr.T))
            sat_diags[nelem-1, :, :] = h_inv @ (TD_right * (tr @ tr.T) - sD_right * Dgk.T @ tr.T + sN_right * tr @ Dgk
                                                + T1 * (tl @ tl.T) + T5v * (tr @ tl.T)
                                                + T3v * (tl @ Dgv) + T2v * (Dgv.T @ tl.T))

            # build SAT matrix
            offset = n
            aux = np.empty((0, offset), int)
            aux2 = np.empty((0, offset * 2), int)

            sat_0 = sp.linalg.block_diag(*sat_diags)
            sat_p1 = sp.linalg.block_diag(aux, *sat_diags_p1[1:nelem, :, :], aux.T)
            sat_m1 = sp.linalg.block_diag(aux.T, *sat_diags_m1[0:nelem - 1, :, :], aux)
            sat_p2 = sp.linalg.block_diag(aux2, *sat_diags_p2[2:nelem, :, :], aux2.T)
            sat_m2 = sp.linalg.block_diag(aux2.T, *sat_diags_m2[0:nelem - 2, :, :], aux2)

            if nelem < 3:
                if nelem < 2:
                    sat_p1 = 0
                    sat_p2 = 0
                    sat_m1 = 0
                    sat_m2 = 0
                else:
                    sat_p2 = 0
                    sat_m2 = 0

        else:
            sat_diags = h_inv @ (sD_left*TD_left * (tl @ tl.T) - sD_left * Dgv.T @ tl.T
                                          + sN_left * tl @ Dgv)
            sat_diags = sat_diags+ h_inv @ (sD_right* TD_right * (tr @ tr.T) - sD_right * Dgk.T @ tr.T
                                                               + sN_right * tr @ Dgk)
            sat_0 = sat_diags

        sat = sat_m2 + sat_m1 + sat_0 + sat_p1 + sat_p2
        sI = sparse.csr_matrix(sat)

        # construct component of the right hand side that comes from the SATs
        fD_left = 0*tl
        fD_right = 0*tr
        fN_left = 0*tl
        fN_right =0*tr

        if uD_left != None:
            fD_left = h_inv @ (-TD_left *tl* uD_left + sD_left* Dgv.T *uD_left)
        if uD_right != None:
            fD_right = h_inv @ (-TD_right*tr*uD_right  + sD_right*Dgk.T *uD_right)
        if uN_left!=None:
            fN_left = h_inv @ (-sN_left * tl * uN_left)
        if uN_right != None:
            fN_right = h_inv @(-sN_right * tr * uN_right)

        fB = np.zeros((n, nelem))
        if nelem >=2:
            fB[:,0] =  fD_left.flatten() + fN_left.flatten()
            fB[:,nelem-1] = fD_right.flatten() + fN_right.flatten()
        else:
            fB[:, 0] = fD_left.flatten() + fN_left.flatten()
            fB[:, 0] = fB[:, 0] + fD_right.flatten() + fN_right.flatten()

        # path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\pysbp_results\\advec_diff_results\\figures\\'
        # # print(np.count_nonzero(sat))
        # plt.spy(sat, marker='o', markeredgewidth=0, markeredgecolor='y', markersize=5, markerfacecolor='r')
        # plt.savefig(path + 'sparsity_{}.pdf'.format(flux_type), format='pdf')
        # plt.close()
        # # plt.show()
        return sI, fB, TD_left, TD_right

    @staticmethod
    def advection_sbp_sats_1d_steady(n, nelem, h_mat, tl, tr, rx, a=1, uD_left=None, uD_right=None, flux_type='upwind'):

        if flux_type == 'upwind':
            sigma = 1
        elif flux_type == 'centered':
            sigma = 0
        else:
            raise ("The flux type (SAT type) should be either 'upwind' or 'centered'. ")

        tr = tr.reshape((n, 1))
        tl = tl.reshape((n, 1))

        # scale the matrices (the ones given are for the reference element)
        h_mat = 1 / rx[0, 0] * h_mat
        h_inv = np.linalg.inv(h_mat)
        a_mat = np.diag(a.flatten())

        # construct diagonals of the SAT matrix
        sat_diag = h_inv @ tr @ tr.T @ (-a_mat/2 + sigma*np.abs(a_mat)/2) \
                   - h_inv @ tl @ tl.T @ (-a_mat/2 - sigma*np.abs(a_mat)/2)
        sat_diag_p1 = h_inv @ tr @ tl.T @ (a_mat/2 - sigma*np.abs(a_mat)/2)
        sat_diag_m1 = -h_inv @ tl @ tr.T @ (a_mat/2 + sigma*np.abs(a_mat)/2)

        sat_diags = sat_diag.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)
        sat_diags_p1 = sat_diag_p1.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)
        sat_diags_m1 = sat_diag_m1.copy().repeat(nelem).reshape(nelem, n, n, order='F').transpose(0, 2, 1)

        # set boundary conditions
        if a[0] > 0 and sigma == 0:
            sat_diags[nelem - 1, :, :] = - h_inv @ tl @ tl.T @ (-a_mat/2 - sigma*np.abs(a_mat)/2)
        elif a[0] < 0 and sigma == 0:
            sat_diags[0, :, :] = h_inv @ tr @ tr.T @ (-a_mat/2 + sigma*np.abs(a_mat)/2)

        # build SAT matrix
        offset = n
        aux = np.empty((0, offset), int)

        sat_0 = sp.linalg.block_diag(*sat_diags)
        sat_p1 = sp.linalg.block_diag(aux, *sat_diags_p1[1:nelem, :, :], aux.T)
        sat_m1 = sp.linalg.block_diag(aux.T, *sat_diags_m1[0:nelem - 1, :, :], aux)

        sat = sat_m1 + sat_0 + sat_p1
        sI = sparse.csr_matrix(sat)

        # construct component of the right hand side that comes from the SATs
        fD_left = 0 * tl
        fD_right = 0 * tr

        if a[0] > 0:
            fD_left = h_inv @ tl *(a_mat[0,0]/2 + sigma*np.abs(a_mat[0,0])/2)* uD_left
        if a[0] < 0:
            fD_right = -h_inv @ tr *(a_mat[-1,-1]/2 - sigma*np.abs(a_mat[-1,-1])/2)* uD_right

        fB = np.zeros((n, nelem))
        if nelem >= 2:
            fB[:, 0] = fD_left.flatten()
            fB[:, nelem - 1] = fD_right.flatten()
        else:
            fB[:, 0] = fD_left.flatten()
            fB[:, 0] = fB[:, 0] + fD_right.flatten()

        return sI, fB

    # @staticmethod
    # def diffusion_sbp_sat_2d_steady(nnodes, nelem, LxxB, LxyB, LyxB, LyyB, Ds, Dr, H, B1, B2, B3, R1, R2, R3, rx, ry,
    #                                 sx, sy, jac, surf_jac,  nx, ny, etoe, etof, bgrp, bgrpD, bgrpN, flux_type='BR2',
    #                                 uD=None, uN=None):
    #
    #     nfp = int(nx.shape[0] / 3)  # number of nodes per facet, also nfp = p+1
    #     dim = 2
    #     nface = dim + 1
    #     # face id
    #     fid1 = np.arange(0, nfp)
    #     fid2 = np.arange(nfp, 2*nfp)
    #     fid3 = np.arange(2*nfp, 3*nfp)
    #
    #     # boundary group (obtain element number and facet number only)
    #     bgrp = np.vstack(bgrp)[:, 2:4]
    #     # Dirichlet boundary groups by facet
    #     bgrpD1 = bgrpD2 = bgrpD3 = []
    #     if len(bgrpD) != 0:
    #         bgrpD1 = bgrpD[bgrpD[:, 1] == 0, :]
    #         bgrpD2 = bgrpD[bgrpD[:, 1] == 1, :]
    #         bgrpD3 = bgrpD[bgrpD[:, 1] == 2, :]
    #
    #     # get the geometric factors for each element (in rxB, B stands for Block)
    #     rxB = rx.T.reshape(nelem, nnodes, 1)
    #     ryB = ry.T.reshape(nelem, nnodes, 1)
    #     sxB = sx.T.reshape(nelem, nnodes, 1)
    #     syB = sy.T.reshape(nelem, nnodes, 1)
    #
    #     # get volume and surface Jacobians for each elements
    #     jacB = jac.T.reshape(nelem, nnodes, 1)
    #     surf_jac1B = surf_jac[fid1, :].flatten(order='F').reshape(nelem, nfp, 1)
    #     surf_jac2B = surf_jac[fid2, :].flatten(order='F').reshape(nelem, nfp, 1)
    #     surf_jac3B = surf_jac[fid3, :].flatten(order='F').reshape(nelem, nfp, 1)
    #
    #     # get the normal vectors on each facet.
    #     nx1B = nx[fid1, :].flatten(order='F').reshape((nelem, nfp, 1))
    #     ny1B = ny[fid1, :].flatten(order='F').reshape((nelem, nfp, 1))
    #
    #     nx2B = nx[fid2, :].flatten(order='F').reshape((nelem, nfp, 1))
    #     ny2B = ny[fid2, :].flatten(order='F').reshape((nelem, nfp, 1))
    #
    #     nx3B = nx[fid3, :].flatten(order='F').reshape((nelem, nfp, 1))
    #     ny3B = ny[fid3, :].flatten(order='F').reshape((nelem, nfp, 1))
    #
    #     nxB = [nx1B, nx2B, nx3B]
    #     nyB = [ny1B, ny2B, ny3B]
    #
    #     # get the derivative operator on the physical elements and store it for each element
    #     DrB = np.block([Dr] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
    #     DsB = np.block([Ds] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
    #     DxB = rxB * DrB + sxB * DsB
    #     DyB = ryB * DrB + syB * DsB
    #
    #     # np.block([R1] * nelem) is a matrix of size nfp X nelem*nnodes, since python reads row by row first transpose
    #     # it, then reshape it in to 3D array of size nelem X nnodes X nfp, gets the first 3*10 entries and form 10 X 3
    #     # matrix and do that for the second, etc. So we need to transpose 10X3 matrices corresponding to each element
    #     R1B = np.block([R1] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
    #     R2B = np.block([R2] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
    #     R3B = np.block([R3] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
    #
    #     RB = [R1B, R2B, R3B]
    #
    #     # scaled diffusion coefficient
    #     # JLxxB = jacB * LxxB
    #     # JLxyB = jacB * LxyB
    #     # JLyxB = jacB * LyxB
    #     # JLyyB = jacB * LyyB
    #
    #     # get derivative operator on each facet
    #     Dgk1B = (nx1B * R1B @ (LxxB @ DxB + LxyB @ DyB) + ny1B * R1B @ (LyxB @ DxB + LyyB @ DyB))
    #     Dgk2B = (nx2B * R2B @ (LxxB @ DxB + LxyB @ DyB) + ny2B * R2B @ (LyxB @ DxB + LyyB @ DyB))
    #     Dgk3B = (nx3B * R3B @ (LxxB @ DxB + LxyB @ DyB) + ny3B * R3B @ (LyxB @ DxB + LyyB @ DyB))
    #
    #     Dgk = [Dgk1B, Dgk2B, Dgk3B]
    #
    #     # get volume norm matrix and its inverse on physical elements
    #     HB = jacB*np.block([H] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
    #     HB_inv = np.linalg.inv(HB)
    #
    #     # get surface norm matrix for each facet of each element
    #     BB1 = (surf_jac1B * np.block([B1] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1))
    #     BB2 = (surf_jac2B/(np.sqrt(2)) * np.block([B2] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1))
    #     BB3 = (surf_jac3B * np.block([B3] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1))
    #
    #     BB = [BB1, BB2, BB3]
    #
    #     # compute the length of each face
    #     face_size = np.zeros((nelem, nface))
    #     for elem in range(0, nelem):
    #         for face in range(0, nface):
    #             face_size[elem, face] = np.sum(BB[face][elem, :, :])
    #
    #     # compute the total length of Dirichlet boundary faces for each element
    #     bndry_face_size = np.zeros((nelem, 1))
    #     bndry_face_size[bgrpD1[:, 0]] += face_size[bgrpD1[:, 0], bgrpD1[:, 1]].reshape(-1, 1)
    #     bndry_face_size[bgrpD2[:, 0]] += face_size[bgrpD2[:, 0], bgrpD2[:, 1]].reshape(-1, 1)
    #     bndry_face_size[bgrpD3[:, 0]] += face_size[bgrpD3[:, 0], bgrpD3[:, 1]].reshape(-1, 1)
    #
    #     # compute the total length of the interior edges of each element
    #     elem_face_size = np.sum(face_size, axis=1).reshape(-1, 1) - bndry_face_size
    #
    #     # calculate face weight \alpha_{\gamma k} for each face
    #     face_wt = np.zeros((nelem, nface))
    #     for elem in range(0, nelem):
    #         for face in range(0, nface):
    #             face_wt[elem, face] = face_size[elem, face]/(elem_face_size[elem] + 2*bndry_face_size[elem])
    #
    #     # change face weight for Dirichlet boundary faces
    #     for i in range(0, len(bgrpD)):
    #         elem = bgrpD[i, 0]
    #         face = bgrpD[i, 1]
    #         face_wt[elem, face] = 2*face_size[elem, face]/(elem_face_size[elem] + 2*bndry_face_size[elem])
    #
    #     # invert face weight (as we need to multiply by 1/alpha)
    #     face_wt = (1/face_wt)
    #
    #     # face weight by face number repeated for each node on the face
    #     face_wt1B = np.block(np.repeat(face_wt[:, 0], nfp)).reshape(nx1B.shape)
    #     face_wt2B = np.block(np.repeat(face_wt[:, 1], nfp)).reshape(nx1B.shape)
    #     face_wt3B = np.block(np.repeat(face_wt[:, 2], nfp)).reshape(nx1B.shape)
    #
    #     face_wtB = [face_wt1B, face_wt2B, face_wt3B]
    #
    #     # calculate Upsilon in Eq.(74) in my notes
    #     # Ugk1B = ((nx1B * R1B) @ (HB_inv @ (JLxxB*rxB + JLyyB*ryB)) @ ((nx1B * R1B).transpose(0, 2, 1))
    #     #         +(ny1B * R1B) @ (HB_inv @ (JLyyB*syB + JLxxB*sxB)) @ ((ny1B * R1B).transpose(0, 2, 1)))
    #     # Ugk2B = ((nx2B * R2B) @ (HB_inv @ (JLxxB*rxB + JLyyB*ryB)) @ ((nx2B * R2B).transpose(0, 2, 1))
    #     #         +(ny2B * R2B) @ (HB_inv @ (JLyyB*syB + JLxxB*sxB)) @ ((ny2B * R2B).transpose(0, 2, 1)))
    #     # Ugk3B = ((nx3B * R3B) @ (HB_inv @ (JLxxB*rxB + JLyyB*ryB)) @ ((nx3B * R3B).transpose(0, 2, 1))
    #     #         +(ny3B * R3B) @ (HB_inv @ (JLyyB*syB + JLxxB*sxB)) @ ((ny3B * R3B).transpose(0, 2, 1)))
    #
    #     Ugk1B = ((nx1B * R1B) @ (HB_inv @ LxxB) @ ((nx1B * R1B).transpose(0, 2, 1))
    #             +(ny1B * R1B) @ (HB_inv @ LyyB) @ ((ny1B * R1B).transpose(0, 2, 1)))
    #     Ugk2B = ((nx2B * R2B) @ (HB_inv @ LxxB) @ ((nx2B * R2B).transpose(0, 2, 1))
    #             +(ny2B * R2B) @ (HB_inv @ LyyB) @ ((ny2B * R2B).transpose(0, 2, 1)))
    #     Ugk3B = ((nx3B * R3B) @ (HB_inv @ LxxB) @ ((nx3B * R3B).transpose(0, 2, 1))
    #             +(ny3B * R3B) @ (HB_inv @ LyyB) @ ((ny3B * R3B).transpose(0, 2, 1)))
    #
    #     Ugk = [Ugk1B, Ugk2B, Ugk3B]
    #
    #     # SAT coefficients for different methods
    #     eta = 1; signT2 = -1; etaD = 1       # BR2 method
    #     # eta = 0; signT2 = 1;  etaD = 1        # BO
    #
    #     # facet 1
    #     T2gk1B = signT2/2*BB1
    #     T3gk1B = 1/2*BB1
    #     T4gk1B = 0*BB1
    #
    #     # facet 2
    #     T2gk2B = signT2/2*BB2
    #     T3gk2B = 1/2*BB2
    #     T4gk2B = 0*BB2
    #
    #     # facet 3
    #     T2gk3B = signT2/2*BB3
    #     T3gk3B = 1/2*BB3
    #     T4gk3B = 0*BB3
    #
    #     # calcualte the T1gk matrix
    #     T1gk1B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))     # T1gk at facet 1
    #     T1gk2B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))     # T1gk at facet 2
    #     T1gk3B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))     # T1gk at facet 3
    #
    #     # T1gk for BR2 method
    #     for elem in range(0, nelem):
    #         for face in range(0, nface):
    #             nbr_elem = etoe[elem, face]
    #             nbr_face = etof[elem, face]
    #             if face==0:
    #                 T1gk1B[elem] = eta*face_wtB[face][elem]/4 * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]
    #                              + (BB[nbr_face][nbr_elem] @ Ugk[nbr_face][nbr_elem] @ BB[nbr_face][nbr_elem]))
    #             elif face==1:
    #                 T1gk2B[elem] = eta*face_wtB[face][elem]/4 * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]
    #                              + (BB[nbr_face][nbr_elem] @ Ugk[nbr_face][nbr_elem] @ BB[nbr_face][nbr_elem]))
    #             elif face==2:
    #                 T1gk3B[elem] = eta*face_wtB[face][elem]/4 * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]
    #                              + (BB[nbr_face][nbr_elem] @ Ugk[nbr_face][nbr_elem] @ BB[nbr_face][nbr_elem]))
    #
    #     # calculate the TDg matrix (the SAT coefficient at Dirichlet boundaries)
    #     TDgk1B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))
    #     TDgk2B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))
    #     TDgk3B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))
    #
    #     for i in range(0, len(bgrpD)):
    #         if bgrpD[i, 1] == 0:
    #             elem = bgrpD[i, 0]
    #             face = bgrpD[i, 1]
    #             TDgk1B[elem] = (etaD*face_wtB[face][elem]*(BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]))
    #         if bgrpD[i, 1] == 1:
    #             elem = bgrpD[i, 0]
    #             face = bgrpD[i, 1]
    #             TDgk2B[elem] = (etaD*face_wtB[face][elem]*(BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]))
    #         if bgrpD[i, 1] == 2:
    #             elem = bgrpD[i, 0]
    #             face = bgrpD[i, 1]
    #             TDgk3B[elem] = (etaD*face_wtB[face][elem]*(BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]))
    #
    #     # put coefficinets in a list to access them by facet number, i.e., facet 1, 2, 3 --> 0, 1, 2
    #     T1gk = [T1gk1B, T1gk2B, T1gk3B]
    #     T2gk = [T2gk1B, T2gk2B, T2gk3B]
    #     T3gk = [T3gk1B, T3gk2B, T3gk3B]
    #     T4gk = [T4gk1B, T4gk2B, T4gk3B]
    #     TDgk = [TDgk1B, TDgk2B, TDgk3B]
    #
    #     # construct a block matrix to hold all the interface SATs
    #     sI = (np.block(np.zeros((nelem*nnodes, nelem*nnodes)))).reshape((nelem, nelem, nnodes, nnodes))
    #
    #     sI_diag = (HB_inv) @ ((np.block([R1B.transpose(0, 2, 1), Dgk1B.transpose(0, 2, 1)])
    #                         @ np.block([[T1gk1B, T3gk1B], [T2gk1B, T4gk1B]]) @ np.block([[R1B], [Dgk1B]]))
    #                         + (np.block([R2B.transpose(0, 2, 1), Dgk2B.transpose(0, 2, 1)])
    #                         @ np.block([[T1gk2B, T3gk2B], [T2gk2B, T4gk2B]]) @ np.block([[R2B], [Dgk2B]]))
    #                         + (np.block([R3B.transpose(0, 2, 1), Dgk3B.transpose(0, 2, 1)])
    #                         @ np.block([[T1gk3B, T3gk3B], [T2gk3B, T4gk3B]]) @ np.block([[R3B], [Dgk3B]])))
    #
    #     # add the diagonals of the SAT matrix
    #     for i in range(0, nelem):
    #         sI[i, i] += sI_diag[i]
    #
    #     # subtract interface SATs added at boundary facets
    #     for i in range(0, bgrp.shape[0]):
    #         elem = bgrp[i, 0]
    #         face = bgrp[i, 1]
    #         sI[elem, elem] += -1*HB_inv[elem] @ (np.block([RB[face][elem].T, Dgk[face][elem].T])
    #                                              @ np.block([[T1gk[face][elem], T3gk[face][elem]],
    #                                                          [T2gk[face][elem], T4gk[face][elem]]])
    #                                              @ np.block([[RB[face][elem]], [Dgk[face][elem]]]))
    #
    #     # add the interface SATs at the neighboring elements
    #     for elem in range(0, nelem):
    #         if elem != etoe[elem, 0]:
    #             # facet 1
    #             face = 0
    #             nbr_elem = etoe[elem, 0]
    #             nbr_face = etof[elem, 0]
    #             sI[elem, nbr_elem] += (HB_inv[elem] @ (np.block([RB[face][elem].T, Dgk[face][elem].T])
    #                                                   @ np.block([[T1gk[face][elem], T3gk[face][elem]],
    #                                                               [T2gk[face][elem], T4gk[face][elem]]])
    #                                                   @ (np.block([[-1*RB[nbr_face][nbr_elem]],
    #                                                               [Dgk[nbr_face][nbr_elem]]]))))
    #         if elem != etoe[elem, 1]:
    #             # facet 2
    #             face = 1
    #             nbr_elem = etoe[elem, 1]
    #             nbr_face = etof[elem, 1]
    #             sI[elem, nbr_elem] += (HB_inv[elem] @ (np.block([RB[face][elem].T, Dgk[face][elem].T])
    #                                                         @ np.block([[T1gk[face][elem], T3gk[face][elem]],
    #                                                                     [T2gk[face][elem], T4gk[face][elem]]])
    #                                                         @ (np.block([[-1*RB[nbr_face][nbr_elem]],
    #                                                                     [Dgk[nbr_face][nbr_elem]]]))))
    #         if elem != etoe[elem, 2]:
    #             # facet 3
    #             face = 2
    #             nbr_elem = etoe[elem, 2]
    #             nbr_face = etof[elem, 2]
    #             sI[elem, nbr_elem] += (HB_inv[elem] @ (np.block([RB[face][elem].T, Dgk[face][elem].T])
    #                                                         @ np.block([[T1gk[face][elem], T3gk[face][elem]],
    #                                                                     [T2gk[face][elem], T4gk[face][elem]]])
    #                                                         @ (np.block([[-1*RB[nbr_face][nbr_elem]],
    #                                                                     [Dgk[nbr_face][nbr_elem]]]))))
    #
    #     # if not given, construct the forcing terms that go to right hand side
    #     if uD is None:
    #         uD = np.zeros((nface * nfp, nelem))
    #     if uN is None:
    #         uN = np.zeros((nface * nfp, nelem))
    #
    #     elemlist =[]
    #     facelist =[]
    #     satlist=[]
    #     # add Dirichlet boundary SATs (and construct sD matrix to obtain the Dirichlet SAT contribution  to the RHS)
    #     for i in range(0, len(bgrpD)):
    #         elem = bgrpD[i, 0]
    #         face = bgrpD[i, 1]
    #         sI[elem, elem] += HB_inv[elem] @ (np.block([RB[face][elem].T, Dgk[face][elem].T])
    #                                           @ np.block([[TDgk[face][elem]], [-1*BB[face][elem]]])
    #                                           @ RB[face][elem])
    #
    #     sD = np.block(np.zeros((nelem * nnodes, nelem * nfp * nface))).reshape((nelem, nelem * nface, nnodes, nfp))
    #     for i in range(0, len(bgrpD)):
    #         elem = bgrpD[i, 0]
    #         face = bgrpD[i, 1]
    #         sD[elem, nface*elem+face] += -1*HB_inv[elem] @ (np.block([RB[face][elem].T, Dgk[face][elem].T])
    #                                                         @ np.block([[TDgk[face][elem]], [-1*BB[face][elem]]]))
    #
    #     sD_mat = (sD.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp * nface)
    #     fD = (sD_mat @ uD.flatten(order="F")).reshape(-1, 1)
    #
    #     sN = np.block(np.zeros((nelem * nnodes, nelem * nfp * nface))).reshape((nelem, nelem * nface, nnodes, nfp))
    #     for i in range(0, len(bgrpN)):
    #         elem = bgrpN[i, 0]
    #         face = bgrpN[i, 1]
    #         sN[elem, nface * elem + face] += HB_inv[elem] @ (RB[face][elem].T @ BB[face][elem] @ Dgk[face][elem])
    #
    #     sN_mat = (sN.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp * nface)
    #     fN = (sN_mat @ uN.flatten(order="F")).reshape(-1, 1)
    #
    #     # reshape the 4D array of the SATs into 2D
    #     sI_mat = (sI.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nnodes)
    #     sI_mat = sparse.csr_matrix(sI_mat)
    #
    #     fB = fD + fN
    #     Hg = sparse.block_diag(HB)
    #
    #     return {'sI': sI_mat, 'fB': fB, 'Hg': Hg, 'BB': BB, 'Dgk': Dgk, 'DxB': DxB, 'DyB': DyB, 'nxB': nxB, 'nyB':nyB}


    @staticmethod
    def diffusion_sbp_sat_2d_steady(nnodes, nelem, LxxB, LxyB, LyxB, LyyB, DxB, DyB, HB, BB, RB, rxB, ryB, sxB, syB,
                                    jacB, surf_jacB, nxB, nyB, etoe, etof,  bgrp, bgrpD, bgrpN, flux_type='BR2',
                                    uD=None, uN=None, eqn='primal'):

        nfp = int(nxB[0][0].shape[0])  # number of nodes per facet, also nfp = p+1
        dim = 2
        nface = dim + 1
        flux_type = flux_type.upper()

        # Dirichlet boundary groups by facet
        bgrpD1 = bgrpD2 = bgrpD3 = []
        if len(bgrpD) != 0:
            bgrpD1 = bgrpD[bgrpD[:, 1] == 0, :]
            bgrpD2 = bgrpD[bgrpD[:, 1] == 1, :]
            bgrpD3 = bgrpD[bgrpD[:, 1] == 2, :]
        # get all boundary groups
        if len(bgrpN) != 0:
            bgrpB = np.vstack([bgrpD, bgrpN])
        else:
            bgrpB = bgrpD

        # get the normals on each facets
        nx1B = nxB[0]
        nx2B = nxB[1]
        nx3B = nxB[2]

        ny1B = nyB[0]
        ny2B = nyB[1]
        ny3B = nyB[2]

        # get the interpolation/extrapolation matrices on each facet
        R1B = RB[0]
        R2B = RB[1]
        R3B = RB[2]

        # get the surface quadrature weights on each facet
        BB1 = BB[0]
        BB2 = BB[1]
        BB3 = BB[2]

        # get the inverse of the norm matrix
        HB_inv = np.linalg.inv(HB)

        # get derivative operator on each facet
        Dgk1B = (nx1B * R1B @ (LxxB @ DxB + LxyB @ DyB) + ny1B * R1B @ (LyxB @ DxB + LyyB @DyB))
        Dgk2B = (nx2B * R2B @ (LxxB @ DxB + LxyB @ DyB) + ny2B * R2B @ (LyxB @ DxB + LyyB @DyB))
        Dgk3B = (nx3B * R3B @ (LxxB @ DxB + LxyB @ DyB) + ny3B * R3B @ (LyxB @ DxB + LyyB @DyB))

        Dgk = [Dgk1B, Dgk2B, Dgk3B]

        # compute the length of each face
        face_size = np.zeros((nelem, nface))
        for elem in range(0, nelem):
            for face in range(0, nface):
                face_size[elem, face] = np.sum(BB[face][elem, :, :])

        # compute the total length of Dirichlet boundary faces for each element
        bndry_face_size = np.zeros((nelem, 1))
        bndry_face_size[bgrpD1[:, 0]] += face_size[bgrpD1[:, 0], bgrpD1[:, 1]].reshape(-1, 1)
        bndry_face_size[bgrpD2[:, 0]] += face_size[bgrpD2[:, 0], bgrpD2[:, 1]].reshape(-1, 1)
        bndry_face_size[bgrpD3[:, 0]] += face_size[bgrpD3[:, 0], bgrpD3[:, 1]].reshape(-1, 1)

        # compute the total length of the interior edges of each element
        elem_face_size = np.sum(face_size, axis=1).reshape(-1, 1) - bndry_face_size

        # calculate face weight \alpha_{\gamma k} for each face
        face_wt = np.zeros((nelem, nface))
        for elem in range(0, nelem):
            for face in range(0, nface):
                face_wt[elem, face] = face_size[elem, face] / (elem_face_size[elem] + 2 * bndry_face_size[elem])

        # change face weight for Dirichlet boundary faces
        for i in range(0, len(bgrpD)):
            elem = bgrpD[i, 0]
            face = bgrpD[i, 1]
            face_wt[elem, face] = 2 * face_size[elem, face] / (elem_face_size[elem] + 2 * bndry_face_size[elem])

        # invert face weight (as we need to multiply by 1/alpha)
        face_wt = (1 / face_wt)

        # face weight by face number repeated for each node on the face
        face_wt1B = np.block(np.repeat(face_wt[:, 0], nfp)).reshape(nx1B.shape)
        face_wt2B = np.block(np.repeat(face_wt[:, 1], nfp)).reshape(nx1B.shape)
        face_wt3B = np.block(np.repeat(face_wt[:, 2], nfp)).reshape(nx1B.shape)

        face_wtB = [face_wt1B, face_wt2B, face_wt3B]

        # calculate Upsilon in Eq.(74) in my notes
        Ugk1B =  ((nx1B * R1B) @ (HB_inv @ LxxB) @ ((nx1B * R1B).transpose(0, 2, 1))
                + (nx1B * R1B) @ (HB_inv @ LxyB) @ ((ny1B * R1B).transpose(0, 2, 1))
                + (ny1B * R1B) @ (HB_inv @ LyxB) @ ((nx1B * R1B).transpose(0, 2, 1))
                + (ny1B * R1B) @ (HB_inv @ LyyB) @ ((ny1B * R1B).transpose(0, 2, 1)))*face_wt1B
        Ugk2B =  ((nx2B * R2B) @ (HB_inv @ LxxB) @ ((nx2B * R2B).transpose(0, 2, 1))
                + (nx2B * R2B) @ (HB_inv @ LxyB) @ ((ny2B * R2B).transpose(0, 2, 1))
                + (ny2B * R2B) @ (HB_inv @ LyxB) @ ((nx2B * R2B).transpose(0, 2, 1))
                + (ny2B * R2B) @ (HB_inv @ LyyB) @ ((ny2B * R2B).transpose(0, 2, 1)))*face_wt2B
        Ugk3B =  ((nx3B * R3B) @ (HB_inv @ LxxB) @ ((nx3B * R3B).transpose(0, 2, 1))
                + (nx3B * R3B) @ (HB_inv @ LxyB) @ ((ny3B * R3B).transpose(0, 2, 1))
                + (ny3B * R3B) @ (HB_inv @ LyxB) @ ((nx3B * R3B).transpose(0, 2, 1))
                + (ny3B * R3B) @ (HB_inv @ LyyB) @ ((ny3B * R3B).transpose(0, 2, 1)))*face_wt3B

        Ugk = [Ugk1B, Ugk2B, Ugk3B]

        # Ugk1B = np.block([np.eye(nfp)] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        # Ugk2B = np.block([np.eye(nfp)] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        # Ugk3B = np.block([np.eye(nfp)] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        # Ugk = [Ugk1B, Ugk2B, Ugk3B]

        # calculate the characteristic mesh size
        hk = 2*jacB
        hc = np.max(hk)

        # SAT coefficients for different methods
        eta = 1/4; coefT2 = -1/2; coefT3 = 1/2; coefT4 = 0; etaD = 1
        sigma = 1
        if flux_type == 'BR2':
            eta = 1/4; coefT2 = -1/2; coefT3 = 1/2; etaD = 1
        elif flux_type == 'BO':
            eta = 0; coefT2 = 1/2; coefT3 = 1/2; etaD = 1
        elif flux_type == 'NIPG':
            eta = 1; coefT2 = 1/2; coefT3 = 1/2; etaD = 1
        elif flux_type == 'CNG':
            eta = 1/16; coefT2 = 0; coefT3 = 1/2; etaD = 1
        elif flux_type == 'IP':
            NotImplemented('The IP method is not implemented yet.')
        elif flux_type == 'CDG':
            eta = 1/2 ; coefT2 = -1/2; coefT3 = -1/2; etaD = 1; mu = 0
        elif flux_type == 'LDG':
            eta = 1; coefT2 = -1/2; coefT3 = -1/2; coefT5 = 1/16; coefT6 = 1/16; etaD = 1; mu = 0
        elif flux_type == 'BR1':
            eta = 1/2;  coefT2 = -1/2; coefT3 = 1/2; coefT5 = 1/16; coefT6 = 1/16; etaD = 1

        # T4 coefficient
        T4gk1B = coefT4 * BB1
        T4gk2B = coefT4 * BB2
        T4gk3B = coefT4 * BB3

        # calculate the T2 and T3 matrices
        if flux_type != 'LDG' or flux_type != 'CDG':
            # facet 1
            T2gk1B = coefT2 * BB1
            T3gk1B = coefT3 * BB1

            # facet 2
            T2gk2B = coefT2 * BB2
            T3gk2B = coefT3 * BB2

            # facet 3
            T2gk3B = coefT2 * BB3
            T3gk3B = coefT3 * BB3

        if flux_type=='LDG' or flux_type=='CDG':
            # calculate beta_gammak and beta_gammav for CDG and LDG; first, define arbitrary global vector g =[gx, gy]
            gx = 1  # the vector chosen affects the properties of the LDG and CDG method; in particular, when
            gy = 1  # one of the components of the vector is zero, e.g., gx=1, or gy=0. This is because for facets that
            # are parallel to the 0 component, the dot(g, n) = 0 for elements sharing an interface; therefore,
            # the condition betak[elem] + betak[nbr_elem] = 1  is not satisfied

            # initialize the \beta_k and \beta_v vectors
            betak1B = np.zeros(nx1B.shape)
            betak2B = np.zeros(nx1B.shape)
            betak3B = np.zeros(nx1B.shape)
            betav1B = np.zeros(nx1B.shape)
            betav2B = np.zeros(nx1B.shape)
            betav3B = np.zeros(nx1B.shape)

            # calculate beta, betak = 1 if dot(nk, g)>=0 and betak+betav=1. At the boundaries betak = 1.
            for elem in range(0, nelem):
                for j in range(0, nfp):
                    if ((nx1B[elem][j] * gx + ny1B[elem][j] * gy) + 0.0) >= 0:
                        betak1B[elem][j] += 1
                    else:
                        betav1B[elem][j] += 1

                    if ((nx2B[elem][j] * gx + ny2B[elem][j] * gy) + 0.0) >= 0:
                        betak2B[elem][j] += 1
                    else:
                        betav2B[elem][j] += 1

                    if ((nx3B[elem][j] * gx + ny3B[elem][j] * gy) + 0.0) >= 0:
                        betak3B[elem][j] += 1
                    else:
                        betav3B[elem][j] += 1

            betak = [betak1B, betak2B, betak3B]
            betav = [betav1B, betav2B, betav3B]

            # ensure betak is 1 at the boundaries
            for i in range(0, len(bgrpD)):
                elem = bgrpD[i, 0]
                face = bgrpD[i, 1]
                betak[face][elem][:] = 0
                betav[face][elem][:] = 1

            for i in range(0, len(bgrpN)):
                elem = bgrpN[i, 0]
                face = bgrpN[i, 1]
                betak[face][elem][:] = 1
                betav[face][elem][:] = 0

            T2gk = [BB1*0, BB2*0, BB3*0]
            T3gk = [BB1*0, BB2*0, BB3*0]
            for elem in range(0, nelem):
                for face in range(0, nface):
                    nbr_elem = etoe[elem, face]
                    if nbr_elem != elem:
                        nbr_face = etof[elem, face]
                        T2gk[face][elem][:] = coefT2 * (betak[face][elem] - betak[nbr_face][nbr_elem] + 1) * BB[face][elem]
                        T3gk[face][elem][:] = coefT3 * (betak[face][elem] - betak[nbr_face][nbr_elem] - 1) * BB[face][elem]

            T2gk1B = T2gk[0]
            T2gk2B = T2gk[1]
            T2gk3B = T2gk[2]
            T3gk1B = T3gk[0]
            T3gk2B = T3gk[1]
            T3gk3B = T3gk[2]

        # calcualte the T1gk and T5gk matrices
        T1gk1B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T1gk at facet 1
        T1gk2B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T1gk at facet 2
        T1gk3B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T1gk at facet 3

        # T1gk for all SATs except SIPG and NIPG methods
        if flux_type=='BR1' or flux_type=='BR2' or flux_type=='CNG':
            for elem in range(0, nelem):
                for face in range(0, nface):
                    nbr_elem = etoe[elem, face]
                    nbr_face = etof[elem, face]
                    # flip facet norm and SAT coefficients (because abutting facets have opposite orientations)
                    BB_nbr = np.flipud(np.fliplr(BB[nbr_face][nbr_elem]))
                    Ugk_nbr = np.flipud(np.fliplr(Ugk[nbr_face][nbr_elem]))

                    if face == 0:
                        T1gk1B[elem] = eta * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem] + (BB_nbr @ Ugk_nbr @ BB_nbr))
                    elif face == 1:
                        T1gk2B[elem] = eta * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem] + (BB_nbr @ Ugk_nbr @ BB_nbr))
                    elif face == 2:
                        T1gk3B[elem] = eta * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem] + (BB_nbr @ Ugk_nbr @ BB_nbr))

        if flux_type=='BR1' or flux_type=='LDG':
            # T5ek for BR1 and LDG SATs
            Ugxik12B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # Upsilon_{gamma \xi k} from facet 1 to facet 2
            Ugxik13B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # Upsilon_{gamma \xi k} from facet 1 to facet 3
            Ugxik21B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # Upsilon_{gamma \xi k} from facet 2 to facet 1
            Ugxik23B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # Upsilon_{gamma \xi k} from facet 2 to facet 3
            Ugxik31B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # Upsilon_{gamma \xi k} from facet 3 to facet 1
            Ugxik32B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # Upsilon_{gamma \xi k} from facet 3 to facet 2
            Ugxik1 = [[], Ugxik12B, Ugxik13B]
            Ugxik2 = [Ugxik21B, [], Ugxik23B]
            Ugxik3 = [Ugxik31B, Ugxik32B, []]
            Ugxik = [Ugxik1, Ugxik2, Ugxik3]

            T5ek12B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T5ek from facet 1 to facet 2
            T5ek13B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T5ek from facet 1 to facet 3
            T5ek21B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T5ek from facet 2 to facet 1
            T5ek23B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T5ek from facet 2 to facet 3
            T5ek31B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T5ek from facet 3 to facet 1
            T5ek32B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))  # T5ek from facet 3 to facet 2
            T5ek1 = [[], T5ek12B, T5ek13B]
            T5ek2 = [T5ek21B, [], T5ek23B]
            T5ek3 = [T5ek31B, T5ek32B, []]
            T5ek = [T5ek1, T5ek2, T5ek3]

            for elem in range(0, nelem):
                for face in range(0, nface):
                    face_other = np.asarray(list({0, 1, 2}.difference({face})))
                    for i in range(0, nface-1):
                        Ugxik[face][face_other[i]][elem] = (np.diag(nxB[face][elem].flatten()) @ RB[face][elem] @ HB_inv[elem] @ LxxB[elem]
                                            @ RB[face_other[i]][elem].T @ np.diag(nxB[face_other[i]][elem].flatten())
                                            + np.diag(nxB[face][elem].flatten()) @ RB[face][elem] @ HB_inv[elem] @ LxyB[elem]
                                            @ RB[face_other[i]][elem].T @ np.diag(nyB[face_other[i]][elem].flatten())
                                            + np.diag(nyB[face][elem].flatten()) @ RB[face][elem] @ HB_inv[elem] @ LyxB[elem]
                                            @ RB[face_other[i]][elem].T @ np.diag(nxB[face_other[i]][elem].flatten())
                                            + np.diag(nyB[face][elem].flatten()) @ RB[face][elem] @ HB_inv[elem] @ LyyB[elem]
                                            @ RB[face_other[i]][elem].T @ np.diag(nyB[face_other[i]][elem].flatten()))

                        T5ek[face][face_other[i]][elem] = (BB[face][elem] @ Ugxik[face][face_other[i]][elem] @ BB[face_other[i]][elem])

        if flux_type=='LDG' or flux_type=='CDG':
            for elem in range(0, nelem):
                for face in range(0, nface):
                    nbr_elem = etoe[elem, face]
                    nbr_face = etof[elem, face]
                    # flip facet norm and SAT coefficients (because abutting facets have opposite orientations)
                    BB_nbr = np.flipud(np.fliplr(BB[nbr_face][nbr_elem]))
                    Ugk_nbr = np.flipud(np.fliplr(Ugk[nbr_face][nbr_elem]))
                    betak_nbr = np.flipud(betav[nbr_face][nbr_elem])

                    if face == 0:
                        T1gk1B[elem] = eta * ((BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]
                                        + BB_nbr @ Ugk_nbr @ BB_nbr)
                                        + (betak[face][elem] - betak_nbr) * (BB[face][elem] @ Ugk[face][elem]
                                        @ BB[face][elem] - BB_nbr @ Ugk_nbr @ BB_nbr)) + mu*BB[face][elem]
                    elif face == 1:
                        T1gk2B[elem] = eta * ((BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]
                                        + BB_nbr @ Ugk_nbr @ BB_nbr)
                                        + (betak[face][elem] - betak_nbr) * (BB[face][elem] @ Ugk[face][elem]
                                        @ BB[face][elem] - BB_nbr @ Ugk_nbr @ BB_nbr)) + mu*BB[face][elem]
                    elif face == 2:
                        T1gk3B[elem] = eta * ((BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]
                                        + BB_nbr @ Ugk_nbr @ BB_nbr)
                                        + (betak[face][elem] - betak_nbr) * (BB[face][elem] @ Ugk[face][elem]
                                        @ BB[face][elem] - BB_nbr @ Ugk_nbr @ BB_nbr)) + mu*BB[face][elem]

        if flux_type == 'NIPG' :
            T1gk1B = (eta / hc) * BB1
            T1gk2B = (eta / hc) * BB2
            T1gk3B = (eta / hc) * BB3

        # calculate the TDg matrix (the SAT coefficient at Dirichlet boundaries)
        TDgk1B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))
        TDgk2B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))
        TDgk3B = np.block(np.zeros((nelem, nfp, nfp))).reshape((nelem, nfp, nfp))

        for i in range(0, len(bgrpD)):
            if bgrpD[i, 1] == 0:
                elem = bgrpD[i, 0]
                face = bgrpD[i, 1]
                TDgk1B[elem] = (etaD * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]))
            if bgrpD[i, 1] == 1:
                elem = bgrpD[i, 0]
                face = bgrpD[i, 1]
                TDgk2B[elem] = (etaD * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]))
            if bgrpD[i, 1] == 2:
                elem = bgrpD[i, 0]
                face = bgrpD[i, 1]
                TDgk3B[elem] = (etaD * (BB[face][elem] @ Ugk[face][elem] @ BB[face][elem]))

        # put coefficients in a list to access them by facet number, i.e., facet 1, 2, 3 --> 0, 1, 2
        T1gk = [T1gk1B, T1gk2B, T1gk3B]
        T2gk = [T2gk1B, T2gk2B, T2gk3B]
        T3gk = [T3gk1B, T3gk2B, T3gk3B]
        T4gk = [T4gk1B, T4gk2B, T4gk3B]
        TDgk = [TDgk1B, TDgk2B, TDgk3B]

        # construct the SAT matrix
        sI = sparse.lil_matrix((nnodes*nelem, nnodes*nelem), dtype=np.float64)
        # sI2 = sparse.lil_matrix((nnodes * nelem, nnodes * nelem), dtype=np.float64)

        if eqn=='primal':
            for elem in range(0, nelem):
                for face in range(0, nface):
                    if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpB):
                        sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += HB_inv[elem] \
                                                                @ (RB[face][elem].T @ T1gk[face][elem] @ RB[face][elem]\
                                                                + RB[face][elem].T @ T3gk[face][elem] @ Dgk[face][elem]\
                                                                + Dgk[face][elem].T @ T2gk[face][elem] @ RB[face][elem]\
                                                                + Dgk[face][elem].T @ T4gk[face][elem] @ Dgk[face][elem])
            for elem in range(0, nelem):
                for face in range(0, nface):
                    face_other = np.asarray(list({0, 1, 2}.difference({face})))
                    elem_nbr = etoe[elem, face]
                    face_gamma_nbr = etof[elem, face]

                    # SAT terms from neighboring elements -- i.e., the subtracted part in terms containing (uk - uv)
                    if elem_nbr != elem:
                        nbr_face = etof[elem, face]
                        sI[elem*nnodes:(elem+1)*nnodes, elem_nbr*nnodes:(elem_nbr+1)*nnodes] += HB_inv[elem]\
                                                @ (-RB[face][elem].T @ T1gk[face][elem] @ np.flipud(RB[nbr_face][elem_nbr])
                                                + RB[face][elem].T @ T3gk[face][elem] @ np.flipud(Dgk[nbr_face][elem_nbr])
                                                - Dgk[face][elem].T @ T2gk[face][elem] @ np.flipud(RB[nbr_face][elem_nbr])
                                                + Dgk[face][elem].T @ T4gk[face][elem] @ np.flipud(Dgk[nbr_face][elem_nbr]))

                    # add BR1 sat terms to interior facets
                    if flux_type == 'BR1':
                        for i in range(0, nface - 1):
                            if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpB) and \
                                    not any(np.array_equal(np.array([elem, face_other[i]]), rowB) for rowB in bgrpB):

                                # add T5 term
                                sI[elem * nnodes:(elem + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (coefT5*T5ek[face][face_other[i]][elem])
                                                       @ RB[face_other[i]][elem])
                                # add T6 term (note T5 = -T6, so we have -coefT6 here)
                                sI[elem_nbr * nnodes:(elem_nbr + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T @ np.fliplr(np.flipud(-coefT5*T5ek[face][face_other[i]][elem]))
                                                           @ (np.flipud(RB[face_other[i]][elem])))

                                nbr_elem_other = etoe[elem, face_other[i]]
                                nbr_face_other = etof[elem, face_other[i]]
                                # subtract T5
                                sI[elem*nnodes:(elem+1)*nnodes, nbr_elem_other*nnodes:(nbr_elem_other+1)*nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (-coefT5*T5ek[face][face_other[i]][elem])
                                                        @ np.flipud(RB[nbr_face_other][nbr_elem_other]))
                                # subtract T6 (again T5 = -T6, so no negative on coefT6 here)
                                sI[elem_nbr*nnodes:(elem_nbr+1)*nnodes, nbr_elem_other*nnodes:(nbr_elem_other+1)*nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T @ np.fliplr(np.flipud(coefT5*T5ek[face][face_other[i]][elem]))
                                                        @ (RB[nbr_face_other][nbr_elem_other]))

                                #------------------
                                # eigtest = np.linalg.eigvals(face_wtB[face_other[i]][elem]*Ugk[face][elem] - Ugxik[face][face_other[i]][elem] \
                                #         @ np.linalg.inv(1/face_wtB[face_other[i]][elem]*Ugk[face_other[i]][elem]) @ Ugxik[face_other[i]][face][elem])
                                # if any(eigtest <= 0):
                                #     print(eigtest)
                                #------------------

                    # add LDG sat terms to interior facets
                    if flux_type == 'LDG':
                        for i in range(0, nface - 1):
                            if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpB) and \
                                    not any(np.array_equal(np.array([elem, face_other[i]]), rowB) for rowB in bgrpB):

                                # calculate coefficient based on the \betak and \betav values at the facets
                                T5 = coefT5 * (1 + betak[face][elem] - np.flipud(betak[face_gamma_nbr][elem_nbr])) \
                                     * (1 + betak[face_other[i]][elem] - np.flipud(betak[etof[elem, face_other[i]]][etoe[elem, face_other[i]]]))
                                T5 = np.diag(T5.flatten())
                                # add T5 term
                                sI[elem * nnodes:(elem + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (T5 @ T5ek[face][face_other[i]][elem])
                                            @ RB[face_other[i]][elem])
                                # add T6 term
                                sI[elem_nbr * nnodes:(elem_nbr + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T
                                                           @ np.fliplr(np.flipud(-T5 @ T5ek[face][face_other[i]][elem]))
                                                           @ (np.flipud(RB[face_other[i]][elem])))

                                nbr_elem_other = etoe[elem, face_other[i]]
                                nbr_face_other = etof[elem, face_other[i]]
                                # subtract T5
                                sI[elem * nnodes:(elem + 1) * nnodes, nbr_elem_other * nnodes:(nbr_elem_other + 1) * nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (-T5 @ T5ek[face][face_other[i]][elem])
                                            @ np.flipud(RB[nbr_face_other][nbr_elem_other]))
                                # subtract T6
                                sI[elem_nbr * nnodes:(elem_nbr + 1) * nnodes, nbr_elem_other * nnodes:(nbr_elem_other + 1) * nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T
                                                           @ np.fliplr(np.flipud(T5 @ T5ek[face][face_other[i]][elem]))
                                                           @ (RB[nbr_face_other][nbr_elem_other]))

        elif eqn=='adjoint':
            for elem in range(0, nelem):
                for face in range(0, nface):
                    if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpB):
                        sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += HB_inv[elem] \
                                                                @ (RB[face][elem].T @ T1gk[face][elem] @ RB[face][elem]\
                                                                + RB[face][elem].T @ (T2gk[face][elem] + BB[face][elem]) @ Dgk[face][elem]\
                                                                + Dgk[face][elem].T @ (T3gk[face][elem] - BB[face][elem]) @ RB[face][elem]\
                                                                + Dgk[face][elem].T @ T4gk[face][elem] @ Dgk[face][elem])

            for elem in range(0, nelem):
                for face in range(0, nface):
                    face_other = np.asarray(list({0, 1, 2}.difference({face})))
                    elem_nbr = etoe[elem, face]
                    face_gamma_nbr = etof[elem, face]

                    # SAT terms from neighboring elements -- i.e., the subtracted part in terms containing (uk - uv)
                    if elem_nbr != elem:
                        sI[elem*nnodes:(elem+1)*nnodes, elem_nbr*nnodes:(elem_nbr+1)*nnodes] += HB_inv[elem] \
                                                @ (-RB[face][elem].T @ np.flipud(np.fliplr(T1gk[face_gamma_nbr][elem_nbr])) @ np.flipud(RB[face_gamma_nbr][elem_nbr])
                                                - RB[face][elem].T @ np.flipud(np.fliplr(T2gk[face_gamma_nbr][elem_nbr])) @ np.flipud(Dgk[face_gamma_nbr][elem_nbr])
                                                + Dgk[face][elem].T @ np.flipud(np.fliplr(T3gk[face_gamma_nbr][elem_nbr])) @ np.flipud(RB[face_gamma_nbr][elem_nbr])
                                                + Dgk[face][elem].T @ np.flipud(np.fliplr(T4gk[face][elem])) @ np.flipud(Dgk[face_gamma_nbr][elem_nbr]))

                    # add BR1 sat terms to interior facets
                    if flux_type == 'BR1':
                        for i in range(0, nface - 1):
                            if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpB) and \
                                    not any(np.array_equal(np.array([elem, face_other[i]]), rowB) for rowB in bgrpB):

                                # add T5 term
                                sI[elem * nnodes:(elem + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (coefT5*T5ek[face][face_other[i]][elem])
                                                       @ RB[face_other[i]][elem])
                                # add T6 term (note T5 = -T6, so we have -coefT6 here)
                                sI[elem_nbr * nnodes:(elem_nbr + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T @ np.fliplr(np.flipud(-coefT5*T5ek[face][face_other[i]][elem]))
                                                           @ (np.flipud(RB[face_other[i]][elem])))

                                nbr_elem_other = etoe[elem, face_other[i]]
                                nbr_face_other = etof[elem, face_other[i]]
                                # subtract T5
                                sI[elem*nnodes:(elem+1)*nnodes, nbr_elem_other*nnodes:(nbr_elem_other+1)*nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (-coefT5*T5ek[face][face_other[i]][elem])
                                                        @ np.flipud(RB[nbr_face_other][nbr_elem_other]))
                                # subtract T6 (again T5 = -T6, so no negative on coefT6 here)
                                sI[elem_nbr*nnodes:(elem_nbr+1)*nnodes, nbr_elem_other*nnodes:(nbr_elem_other+1)*nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T @ np.fliplr(np.flipud(coefT5*T5ek[face][face_other[i]][elem]))
                                                        @ (RB[nbr_face_other][nbr_elem_other]))

                    # add LDG sat terms to interior facets
                    if flux_type == 'LDG':
                        for i in range(0, nface - 1):
                            if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpB) and \
                                    not any(np.array_equal(np.array([elem, face_other[i]]), rowB) for rowB in bgrpB):

                                # calculate coefficient based on the \betak and \betav values at the facets
                                T5 = coefT5 * (1 + betak[face][elem] - np.flipud(betak[face_gamma_nbr][elem_nbr])) \
                                     * (1 + betak[face_other[i]][elem] - np.flipud(betak[etof[elem, face_other[i]]][etoe[elem, face_other[i]]]))
                                T5 = np.diag(T5.flatten())
                                # add T5 term
                                sI[elem * nnodes:(elem + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (T5 @ T5ek[face][face_other[i]][elem])
                                            @ RB[face_other[i]][elem])
                                # add T6 term (note T5 = -T6, so we have -coefT6 here)
                                sI[elem_nbr * nnodes:(elem_nbr + 1) * nnodes, elem * nnodes:(elem + 1) * nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T
                                                           @ np.fliplr(np.flipud(-T5 @ T5ek[face][face_other[i]][elem]))
                                                           @ (np.flipud(RB[face_other[i]][elem])))

                                nbr_elem_other = etoe[elem, face_other[i]]
                                nbr_face_other = etof[elem, face_other[i]]
                                # subtract T5
                                sI[elem * nnodes:(elem + 1) * nnodes,
                                nbr_elem_other * nnodes:(nbr_elem_other + 1) * nnodes] \
                                    += HB_inv[elem] @ (RB[face][elem].T @ (-T5 @ T5ek[face][face_other[i]][elem])
                                            @ np.flipud(RB[nbr_face_other][nbr_elem_other]))
                                # subtract T6 (again T5 = -T6, so no negative on coefT6 here)
                                sI[elem_nbr * nnodes:(elem_nbr + 1) * nnodes, nbr_elem_other * nnodes:(nbr_elem_other + 1) * nnodes] \
                                    += HB_inv[elem_nbr] @ (RB[face_gamma_nbr][elem_nbr].T
                                                           @ np.fliplr(np.flipud(T5 @ T5ek[face][face_other[i]][elem]))
                                                           @ (RB[nbr_face_other][nbr_elem_other]))
        # -------------------------------------------------------------------------------------------------------------
        # Dirichlet boundary condition
        for i in range(0, len(bgrpD)):
            elem = bgrpD[i, 0]
            face = bgrpD[i, 1]
            # add boundary SAT terms
            sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += etaD * HB_inv[elem] \
                                                            @ (RB[face][elem].T @ TDgk[face][elem] @ RB[face][elem]\
                                                            - Dgk[face][elem].T @ BB[face][elem] @ RB[face][elem])

        # Neumann boundary condition
        for i in range(0, len(bgrpN)):
            elem = bgrpN[i, 0]
            face = bgrpN[i, 1]
            # add boundary SAT terms
            sI[elem*nnodes:(elem + 1)*nnodes, elem*nnodes:(elem + 1)*nnodes] += HB_inv[elem] \
                                                            @ (RB[face][elem].T @ BB[face][elem] @ Dgk[face][elem])

        sI_mat = sI.tocsr()

        # if not given, assume homogeneous Dirichlet and Neumann boundary conditions
        if uD is None:
            uD = np.zeros((nface * nfp, nelem))
        if uN is None:
            uN = np.zeros((nface * nfp, nelem))

        # construct SAT matrix that multiplies the Dirichlet boundary vector
        sD = sparse.lil_matrix((nelem*nnodes, nelem*nfp*nface), dtype=np.float64)
        BD = sparse.lil_matrix((nelem*nfp*nface, nelem*nfp*nface), dtype=np.float64)
        Dgk_D = sparse.lil_matrix((nelem*nfp*nface, nelem*nnodes), dtype=np.float64)
        Rgk_D = sparse.lil_matrix((nelem*nfp*nface, nelem*nnodes), dtype=np.float64)
        TDgk_D = sparse.lil_matrix((nelem*nfp*nface, nelem*nfp*nface), dtype=np.float64)
        for i in range(0, len(bgrpD)):
            elem = bgrpD[i, 0]
            face = bgrpD[i, 1]
            sD[elem*nnodes:(elem+1)*nnodes, (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] += HB_inv[elem] @\
                                                                           (-RB[face][elem].T @ TDgk[face][elem]
                                                                            +Dgk[face][elem].T @ BB[face][elem])
            BD[(elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1)), (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] += BB[face][elem]

            Dgk_D[(elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1)), (elem*nnodes):((elem+1)*nnodes)] = Dgk[face][elem]
            Rgk_D[(elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1)), (elem*nnodes):((elem+1)*nnodes)] = RB[face][elem]
            TDgk_D[(elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1)), (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] = TDgk[face][elem]

        sD_mat = sD.tocsr()
        fD = (sD_mat @ uD.flatten(order="F")).reshape(-1, 1)

        # construct SAT matrix that multiplies the Neumann boundary vector
        sN = sparse.lil_matrix((nelem*nnodes, nelem*nfp*nface), dtype=np.float64)
        BN = sparse.lil_matrix((nelem*nfp*nface, nelem*nfp*nface), dtype=np.float64)
        Rgk_N = sparse.lil_matrix((nelem*nfp*nface, nelem*nnodes), dtype=np.float64)
        for i in range(0, len(bgrpN)):
            elem = bgrpN[i, 0]
            face = bgrpN[i, 1]
            sN[elem*nnodes:(elem+1)*nnodes, (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] -= HB_inv[elem] @\
                                                                (RB[face][elem].T @ BB[face][elem])
            BN[(elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1)), (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] += BB[face][elem]

            Rgk_N[(elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1)), (elem*nnodes):((elem+1)*nnodes)] = RB[face][elem]

        sN_mat = sN.tocsr()
        fN = (sN_mat @ uN.flatten(order="F")).reshape(-1, 1)

        fB = fD + fN
        Hg = sparse.block_diag(HB)

        return {'sI': sI_mat, 'fB': fB, 'Hg': Hg, 'Dgk': Dgk, 'BD': BD, 'BN': BN, 'Dgk_D': Dgk_D, 'Rgk_D': Rgk_D,
                'TDgk_D': TDgk_D, 'Rgk_N': Rgk_N}