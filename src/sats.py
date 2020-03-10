import numpy as np
from collections import deque
from scipy import sparse
import scipy as sp
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
            e_mat = np.zeros((n,n))
            e_mat[0, 0] = -1
            e_mat[n-1,n-1] = 1
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
            eta = 2.1
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
            eta = nface
            he = rx[0, 0]
            mu = eta/he
            T1 = mu
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

        if app ==2:
            db_inv = np.linalg.inv(db_mat)
            # A = db_inv.T @ d_mat.T @ h_mat @ d_mat @ db_inv

            # construct the A matrix for CSBP_Mattsson2004 operators
            e_mat = tr @ tr.T - tl @ tl.T

            if b_mat[0,0] !=0:
                A = db_inv.T @ (e_mat @ db_mat - h_mat @ (1/b_mat[0,0] * d2_mat)) @ db_inv
                M = b_mat[0, 0] * np.linalg.pinv(A)
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
                eta = 2
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
            if app == 2:
                T1 = (1/ 8 * (tr.T @ M @ tr + tl.T @ M @ tl))[0, 0]
            else:
                T1 = (1 / 8 * (tr.T @ b_mat @ h_inv @ tr + tl.T @ b_mat @ h_inv @ tl))[0, 0]

            T3k = 1/2
            T3v = 1/2

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
            TD_right =sD_right * 2 * (tr.T @ M @ tr)[0, 0]
        else:
            TD_left = sD_left *2 * (tl.T @ b_mat @ h_inv @ tl)[0, 0]  # Dirichlet boundary flux coefficient at the left bc
            TD_right = sD_right * 2 * (tr.T @ b_mat @ h_inv @ tr)[0, 0]  # Dirichlet boundary flux coefficient at the right bc

        # calculate normal derivative at the interfaces
        if app==2:
            Dgk = nx[0, 1] * (tr.T @ b_mat  @ db_mat)  # D_{\gamma k}
            Dgv = nx[0, 0] * (tl.T @ b_mat @  db_mat)  # D_{\gamma v}
        else:
            Dgk = nx[0, 1]*(tr.T @ b_mat @ d_mat)   # D_{\gamma k}
            Dgv = nx[0, 0]*(tl.T @ b_mat @  d_mat)   # D_{\gamma v}

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
        return sI, fB

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

    @staticmethod
    def diffusion_sbp_sat_2d_steady(nnodes, nelem, LxxB, LxyB, LyxB, LyyB, LB, Ds, Dr, H, B1, B2, B3, R1, R2, R3, rx, ry,
                                    sx, sy, jac, surf_jac,  nx, ny, etoe, etof, bgrp, bgrpD, bgrpN, x, y, xf, yf, flux_type='BR2',
                                    uD=None, uN=None):

        nfp = int(nx.shape[0] / 3)  # number of nodes per facet, also nfp = p+1
        dim = 2
        # boundary group (obtain element number and facet number only)
        bgrp = np.vstack(bgrp)[:, 2:4]
        # Dirichlet boundary groups by facet
        bgrpD1 = bgrpD2 = bgrpD3 = []
        if len(bgrpD) != 0:
            bgrpD1 = bgrpD[bgrpD[:, 1] == 0, :]
            bgrpD2 = bgrpD[bgrpD[:, 1] == 1, :]
            bgrpD3 = bgrpD[bgrpD[:, 1] == 2, :]

        # Neumann boundary groups by facet
        bgrpN1 = bgrpN2 = bgrpN3 = []
        if len(bgrpN) != 0:
            bgrpN1 = bgrpN[bgrpN[:, 1]==0, :]
            bgrpN2 = bgrpN[bgrpN[:, 1]==1, :]
            bgrpN3 = bgrpN[bgrpN[:, 1]==2, :]

        # get the geometric factors for each element (in rxB, B stands for Block)
        rxB = rx.T.reshape(nelem, nnodes, 1)
        ryB = ry.T.reshape(nelem, nnodes, 1)
        sxB = sx.T.reshape(nelem, nnodes, 1)
        syB = sy.T.reshape(nelem, nnodes, 1)

        # get volume and surface Jacobians for each elements
        jacB = jac.T.reshape(nelem, nnodes, 1)
        surf_jac1B = surf_jac[0:nfp, :].flatten(order='F').reshape(nelem, nfp, 1)
        surf_jac2B = surf_jac[nfp:2*nfp, :].flatten(order='F').reshape(nelem, nfp, 1)
        surf_jac3B = surf_jac[2*nfp:3*nfp, :].flatten(order='F').reshape(nelem, nfp, 1)

        # get the normal vectors on each facet.
        nx1B = nx[0:nfp, :].flatten(order='F').reshape(nelem, nfp, 1)
        ny1B = ny[0:nfp, :].flatten(order='F').reshape(nelem, nfp, 1)

        nx2B = nx[nfp:2*nfp, :].flatten(order='F').reshape(nelem, nfp, 1)
        ny2B = ny[nfp:2*nfp, :].flatten(order='F').reshape(nelem, nfp, 1)

        nx3B = nx[2*nfp:3*nfp, :].flatten(order='F').reshape(nelem, nfp, 1)
        ny3B = ny[2*nfp:3*nfp, :].flatten(order='F').reshape(nelem, nfp, 1)

        # get the derivative operator on the physical elements and store it for each element
        DrB = np.block([Dr] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        DsB = np.block([Ds] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        DxB = rxB * DrB + sxB * DsB
        DyB = ryB * DrB + syB * DsB

        # #-------- tests derivavitve operator ------------
        # ee = 0
        # DerX = DxB[ee, :, :]
        # DerY = DyB[ee, :, :]
        # xe = x[:, ee]
        # ye = y[:, ee]
        # kkx = (DerX @ (xe**2 * ye**2))
        # kky = (DerY @ (xe**2 * ye**2))
        # Der_errX = np.max(kkx - 2*xe**1 * ye**2)
        # Der_errY = np.max(kky - 2*xe**2 * ye**1)
        # print(Der_errX)
        # print(Der_errY)
        # # -----------------------------------------------

        # np.block([R1] * nelem) is a matrix of size nfp X nelem*nnodes, since python reads row by row first transpose
        # it, then reshape it in to 3D array of size nelem X nnodes X nfp, gets the first 3*10 entries and form 10 X 3
        # matrix and do that for the second, etc. So we need to transpose 10X3 matrices corresponding to each element
        R1B = np.block([R1] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
        R2B = np.block([R2] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
        R3B = np.block([R3] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)

        RB = [R1B, R2B, R3B]

        # get derivative operator on each facet
        Dgk1B = (nx1B * R1B @ (LxxB @ DxB + LxyB @ DyB) + ny1B * R1B @ (LyxB @ DxB + LyyB @ DyB))
        Dgk2B = (nx2B * R2B @ (LxxB @ DxB + LxyB @ DyB) + ny2B * R2B @ (LyxB @ DxB + LyyB @ DyB))
        Dgk3B = (nx3B * R3B @ (LxxB @ DxB + LxyB @ DyB) + ny3B * R3B @ (LyxB @ DxB + LyyB @ DyB))

        Dgk = [Dgk1B, Dgk2B, Dgk3B]

        # #------------------ test Dgk (with degree p4 operator)-----------
        # # boundary facet nodes by facet number
        # fid1 = np.arange(0, nfp)
        # fid2 = np.arange(nfp, 2*nfp)
        # fid3 = np.arange(2*nfp, 3*nfp)
        # ee = 1
        # Dgk1_errX = Dgk1B[ee, :, :] @ (x[:, ee] ** 4) - 4*nx[fid1, ee] * xf[fid1, ee]**3
        # Dgk2_errX = Dgk2B[ee, :, :] @ (x[:, ee] ** 4) - 4*nx[fid2, ee] * xf[fid2, ee]**3
        # Dgk3_errX = Dgk3B[ee, :, :] @ (x[:, ee] ** 4) - 4*nx[fid3, ee] * xf[fid3, ee]**3
        #
        # Dgk1_errY = Dgk1B[ee, :, :] @ (y[:, ee] ** 4) - 4 * ny[fid1, ee] * yf[fid1, ee] ** 3
        # Dgk2_errY = Dgk2B[ee, :, :] @ (y[:, ee] ** 4) - 4 * ny[fid2, ee] * yf[fid2, ee] ** 3
        # Dgk3_errY = Dgk3B[ee, :, :] @ (y[:, ee] ** 4) - 4 * ny[fid3, ee] * yf[fid3, ee] ** 3
        #
        # Dgk1_errXY = Dgk1B[ee, :, :] @ (y[:, ee]**2 * x[:, ee]**2) - (2*nx[fid1, ee]*xf[fid1,ee] * yf[fid1, ee]**2
        #                                                               + 2*ny[fid1, ee]*yf[fid1,ee] * xf[fid1, ee]**2)
        # Dgk2_errXY = Dgk2B[ee, :, :] @ (y[:, ee]**2 * x[:, ee]**2) - (2*nx[fid2, ee]*xf[fid2,ee] * yf[fid2, ee]**2
        #                                                               + 2*ny[fid2, ee]*yf[fid2,ee] * xf[fid2, ee]**2)
        # Dgk3_errXY = Dgk3B[ee, :, :] @ (y[:, ee]**2 * x[:, ee]**2) - (2*nx[fid3, ee]*xf[fid3,ee] * yf[fid3, ee]**2
        #                                                               + 2*ny[fid3, ee]*yf[fid3,ee] * xf[fid3, ee]**2)
        # #-----------------------------------

        # get volume norm matrix and its inverse on physical elements
        HB = jacB * np.block([H] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        HB_inv = np.linalg.inv(HB)

        # #---------- test -----------
        # HB2 = np.zeros((nelem, nelem, nnodes, nnodes))
        # for i in range(0, nelem):
        #     HB2[i, i, :, :] = HB[i, :, :]
        # HB2 = (HB2.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nnodes)
        #
        # area_err = np.max(np.ones((nnodes*nelem, 1)).T @ HB2 @ np.ones((nnodes*nelem, 1)) - 4) # rectanglular domain on [-1,1], [1,-1],[1, 1], [-1, 1]
        # #---------------------------


        # get surface norm matrix for each facet of each element
        BB1 = surf_jac1B * np.block([B1] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        BB2 = surf_jac2B * np.block([B2] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)
        BB3 = surf_jac3B * np.block([B3] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1)

        BB = [BB1, BB2, BB3]

        # calculate Upsilon in Eq.(74) in my notes
        Ugk1B = (nx1B * R1B) @ (HB_inv @ LxxB) @ (R1B.transpose(0, 2, 1) * nx1B.transpose(0,2,1)) \
                + (nx1B * R1B) @ (HB_inv @ LxyB) @ (R1B.transpose(0, 2, 1) * ny1B.transpose(0,2,1))\
                + (ny1B * R1B) @ (HB_inv @ LyxB) @ (R1B.transpose(0, 2, 1) * nx1B.transpose(0,2,1))\
                + (ny1B * R1B) @ (HB_inv @ LyyB) @ (R1B.transpose(0, 2, 1) * ny1B.transpose(0,2,1))

        Ugk2B = (nx2B * R2B) @ (HB_inv @ LxxB) @ (R2B.transpose(0, 2, 1) * nx2B.transpose(0,2,1)) \
                + (nx2B * R2B) @ (HB_inv @ LxyB) @ (R2B.transpose(0, 2, 1) * ny2B.transpose(0,2,1))\
                + (ny2B * R2B) @ (HB_inv @ LyxB) @ (R2B.transpose(0, 2, 1) * nx2B.transpose(0,2,1))\
                + (ny2B * R2B) @ (HB_inv @ LyyB) @ (R2B.transpose(0, 2, 1) * ny2B.transpose(0,2,1))

        Ugk3B = (nx3B * R3B) @ (HB_inv @ LxxB) @ (R3B.transpose(0, 2, 1) * nx3B.transpose(0,2,1)) \
                + (nx3B * R3B) @ (HB_inv @ LxyB) @ (R3B.transpose(0, 2, 1) * ny3B.transpose(0,2,1))\
                + (ny3B * R3B) @ (HB_inv @ LyxB) @ (R3B.transpose(0, 2, 1) * nx3B.transpose(0,2,1))\
                + (ny3B * R3B) @ (HB_inv @ LyyB) @ (R3B.transpose(0, 2, 1) * ny3B.transpose(0,2,1))

        Ugk = [Ugk1B, Ugk2B, Ugk3B]

        # # get the derivative operator on each element
        # Dr_block = ([Dr] * nelem)  # Dr matrix for every element
        # Ds_block = ([Ds] * nelem)  # Ds matrix for every element
        #
        # # get the derivative on the physical element, we've: Dx = Dr*rx + Ds*sx and  Dy = Dr*ry + Ds*sy
        # DxB = sparse.diags(rx.flatten(order='F')) @ sparse.block_diag(Dr_block) \
        #       + sparse.diags(sx.flatten(order='F')) @ sparse.block_diag(Ds_block)
        # DyB = sparse.diags(ry.flatten(order='F')) @ sparse.block_diag(Dr_block) \
        #       + sparse.diags(sy.flatten(order='F')) @ sparse.block_diag(Ds_block)
        #
        # # get the extrapolation/interpolation matrix in block matrix form
        # R1B = sparse.block_diag([R1] * nelem)
        # R2B = sparse.block_diag([R2] * nelem)
        # R3B = sparse.block_diag([R3] * nelem)
        #
        # # get the normal matrices in block matrix form. Note: due to difference in facet numbering in SBP and DG codes
        # # nx and ny store normals in facet number order of 3, 1, 2)
        # nx1 = sparse.block_diag(nx[nfp:2*nfp, :].flatten(order='F'))
        # ny1 = sparse.block_diag(ny[nfp:2*nfp, :].flatten(order='F'))
        #
        # nx2 = sparse.block_diag(nx[2*nfp:3*nfp, :].flatten(order='F'))
        # ny2 = sparse.block_diag(ny[2*nfp:3*nfp, :].flatten(order='F'))
        #
        # nx3 = sparse.block_diag(nx[0:nfp, :].flatten(order='F'))
        # ny3 = sparse.block_diag(ny[0:nfp, :].flatten(order='F'))
        #
        # # get derivative matrix at the surface of each element
        # Dgk1 = (np.block([nx1, ny1]) @ np.block([[R1B, 0*R1B], [0*R1B, R1B]]) @ (LB @ np.block([[DxB], [DyB]])))[0,0]
        # Dgk2 = (np.block([nx2, ny2]) @ np.block([[R2B, 0*R2B], [0*R2B, R2B]]) @ (LB @ np.block([[DxB], [DyB]])))[0,0]
        # Dgk3 = (np.block([nx3, ny3]) @ np.block([[R3B, 0*R3B], [0*R3B, R3B]]) @ (LB @ np.block([[DxB], [DyB]])))[0,0]

        # SAT coefficients for different methods

        # coefficients for the BR2 method

        eta = 0*(dim + 1)
        # facet 1
        T2gk1B = 1/2*BB1
        T2gv1B = 1/2*BB1
        T3gk1B = 1/2*BB1
        T3gv1B = 1/2*BB1
        T4gk1B = 0*BB1
        T4gv1B = 0*BB1

        # facet 2
        T2gk2B = 1/2*BB2
        T2gv2B = 1/2*BB2
        T3gk2B = 1/2*BB2
        T3gv2B = 1/2*BB2
        T4gk2B = 0*BB2
        T4gv2B = 0*BB2

        # facet 3
        T2gk3B = 1/2*BB3
        T2gv3B = 1/2*BB3
        T3gk3B = 1/2*BB3
        T3gv3B = 1/2*BB3
        T4gk3B = 0*BB3
        T4gv3B = 0*BB3

        # calcualte the T1gk matrix
        T1gk1B = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)     # T1gk at facet 1
        T1gk2B = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)     # T1gk at facet 2
        T1gk3B = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)     # T1gk at facet 3
        # T1gk for BR2 method
        for i in range(0, nelem):
            T1gk1B[i, :, :] = eta / 4 * BB1[i, :, :] @ (Ugk[0][i, :, :] + Ugk[etof[i, 0]][etoe[i, 0], :, :]) @ BB1[i, :, :]
            T1gk2B[i, :, :] = eta / 4 * BB2[i, :, :] @ (Ugk[1][i, :, :] + Ugk[etof[i, 1]][etoe[i, 1], :, :]) @ BB2[i, :, :]
            T1gk3B[i, :, :] = eta / 4 * BB3[i, :, :] @ (Ugk[2][i, :, :] + Ugk[etof[i, 2]][etoe[i, 2], :, :]) @ BB3[i, :, :]

        # calculate the TDg matrix (the SAT coefficient at Dirichlet boundaries)
        # TDg = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)
        TDgk1B = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)
        TDgk2B = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)
        TDgk3B = np.block(np.zeros((nelem, nfp, nfp))).reshape(nelem, nfp, nfp)

        etaD = (dim + 1)
        for i in range(0, len(bgrpD)):
            if bgrpD[i, 1] == 0:
                TDgk1B[bgrpD[i, 0], :, :] = etaD * (BB[bgrpD[i, 1]][bgrpD[i, 0], :, :] @ Ugk[bgrpD[i, 1]][bgrpD[i, 0], :, : ] \
                                        @ BB[bgrpD[i, 1]][bgrpD[i, 0], :, :])
            if bgrpD[i, 1] == 1:
                TDgk2B[bgrpD[i, 0], :, :] = etaD * (BB[bgrpD[i, 1]][bgrpD[i, 0], :, :] @ Ugk[bgrpD[i, 1]][bgrpD[i, 0], :, : ] \
                                        @ BB[bgrpD[i, 1]][bgrpD[i, 0], :, :])
            if bgrpD[i, 1] == 2:
                TDgk3B[bgrpD[i, 0], :, :] = etaD * (BB[bgrpD[i, 1]][bgrpD[i, 0], :, :] @ Ugk[bgrpD[i, 1]][bgrpD[i, 0], :, : ] \
                                        @ BB[bgrpD[i, 1]][bgrpD[i, 0], :, :])
            # TDg[bgrpD[i, 0], :, :] = etaD * (BB[bgrpD[i, 1]][bgrpD[i, 0], :, :] @ Ugk[bgrpD[i, 1]][bgrpD[i, 0], :, : ] \
            #                         @ BB[bgrpD[i, 1]][bgrpD[i, 0], :, :])


        # put coefficinets in a list to access them by facet number, i.e., facet 1, 2, 3 --> 0, 1, 2
        T1gk = [T1gk1B, T1gk2B, T1gk3B]
        T2gk = [T2gk1B, T2gk2B, T2gk3B]
        T3gk = [T3gk1B, T3gk2B, T3gk3B]
        T4gk = [T4gk1B, T4gk2B, T4gk3B]
        TDgk = [TDgk1B, TDgk2B, TDgk3B]

        # construct a block matrix to hold all the interface SATs
        # sI = sparse.bsr_matrix(np.zeros(nelem*nnodes, nelem*nnodes)), blocksize=(nnodes, nnodes))
        sI = (np.block(np.zeros((nelem*nnodes, nelem*nnodes)))).reshape(nelem, nelem, nnodes, nnodes)

        sI_diag = HB_inv @ (np.block([R1B.transpose(0,2,1), Dgk1B.transpose(0,2,1)]) @ np.block([[T1gk1B, T3gk1B], [T2gk1B, T4gk1B]])
                  @ np.block([[R1B], [Dgk1B]])
                  + np.block([R2B.transpose(0,2,1), Dgk2B.transpose(0,2,1)]) @ np.block([[T1gk2B, T3gk2B], [T2gk2B, T4gk2B]])
                  @ np.block([[R2B], [Dgk2B]])
                  + np.block([R3B.transpose(0,2,1), Dgk3B.transpose(0,2,1)]) @ np.block([[T1gk3B, T3gk3B], [T2gk3B, T4gk3B]])
                  @ np.block([[R3B], [Dgk3B]]))

        # add the diagonals of the SAT matrix
        for i in range(0, nelem):
            sI[i, i, :, :] += sI_diag[i, :, :]

        # subtract interface SATs added at boundary facets
        for i in range(0, bgrp.shape[0]):
            sI[bgrp[i, 0], bgrp[i, 0], :, :] -= HB_inv[bgrp[i, 0], :, :]\
                                                @ (np.block([(RB[bgrp[i, 1]].transpose(0, 2, 1))[bgrp[i, 0], :, :],
                                                             (Dgk[bgrp[i, 1]].transpose(0, 2, 1))[bgrp[i, 0], :, :]])
                                                @ np.block([[T1gk[bgrp[i, 1]][bgrp[i, 0], :, :], T3gk[bgrp[i, 1]][bgrp[i, 0], :, :]],
                                                            [T2gk[bgrp[i, 1]][bgrp[i, 0], :, :], T4gk[bgrp[i, 1]][bgrp[i, 0], :, :]]])
                                                @ np.block([[RB[bgrp[i, 1]][bgrp[i, 0], :, :]], [Dgk[bgrp[i, 1]][bgrp[i, 0], :, :]]]))

        for i in range(0, nelem):
            if i != etoe[i, 0]:
                # facet 1
                sI[i, etoe[i, 0], :, :] += HB_inv[i,: :] @ (np.block([R1B.transpose(0,2,1)[i, :, :], Dgk1B.transpose(0,2,1)[i, :, :]])
                                           @ np.block([[T1gk1B[i,:,:], T3gk1B[i,:,:]], [T2gk1B[i,:,:], T4gk1B[i,:,:]]])
                                           @ np.block([[-1*RB[etof[i,0]][etoe[i, 0], :, :]], [Dgk[etof[i,0]][etoe[i, 0], : , : ]]]))
            if i != etoe[i, 1]:
                # facet 2
                sI[i, etoe[i, 1], :, :] += HB_inv[i,: :] @ (np.block([R2B.transpose(0,2,1)[i, :, :], Dgk2B.transpose(0,2,1)[i, :, :]])
                                           @ np.block([[T1gk2B[i,:,:], T3gk2B[i,:,:]], [T2gk2B[i,:,:], T4gk2B[i,:,:]]])
                                           @ np.block([[-1*RB[etof[i,1]][etoe[i, 1], :, :]], [Dgk[etof[i,1]][etoe[i, 1], : , : ]]]))
            if i != etoe[i, 2]:
                # facet 3
                sI[i, etoe[i, 2], :, :] += HB_inv[i,: :] @ (np.block([R3B.transpose(0,2,1)[i, :, :], Dgk3B.transpose(0,2,1)[i, :, :]])
                                           @ np.block([[T1gk3B[i,:,:], T3gk3B[i,:,:]], [T2gk3B[i,:,:], T4gk3B[i,:,:]]])
                                           @ np.block([[-1*RB[etof[i,2]][etoe[i, 2], :, :]], [Dgk[etof[i,2]][etoe[i, 2], : , : ]]]))

        # add Dirichlet boundary SATs (and construct sD matrix to obtain the Dirichlet SAT contribution  to the RHS)
        # there can be more than one facet where boundary condition may be applied to; since sD1 multiplies the values
        # of the Dirichlet bc on facet 1, we can't add all (sD1+sD2+sD3)uD; instead we've to do sD1*uD1+sD2*uD2+sD3*uD3
        sD1 = (np.block(np.zeros((nelem * nnodes, nelem * nfp)))).reshape(nelem, nelem, nnodes, nfp)    # facet 1
        sD2 = (np.block(np.zeros((nelem * nnodes, nelem * nfp)))).reshape(nelem, nelem, nnodes, nfp)    # facet 2
        sD3 = (np.block(np.zeros((nelem * nnodes, nelem * nfp)))).reshape(nelem, nelem, nnodes, nfp)    # facet 3

        for i in range(0, len(bgrpD)):
            sI[bgrpD[i, 0], bgrpD[i, 0], :, :] += HB_inv[bgrpD[i,0],: :] @ (np.block([(RB[bgrpD[i, 1]].transpose(0,2,1))[bgrpD[i, 0], :, :],
                                                           (Dgk[bgrpD[i, 1]].transpose(0,2,1))[bgrpD[i, 0], :, :]]) \
                                                  @ np.block([[TDgk[bgrp[i, 1]][bgrpD[i, 0], :, :]],
                                                              [-1*BB[bgrpD[i, 1]][bgrpD[i, 0], :, :]]]) \
                                                  @ RB[bgrpD[i, 1]][bgrpD[i, 0], :, :])

        if len(bgrpD1) != 0:
           for i in range(0, len(bgrpD1)):
                sD1[bgrpD1[i, 0], bgrpD1[i, 0], :, :] += -1*HB_inv[bgrpD1[i,0],: :] @ (np.block([(RB[bgrpD1[i, 1]].transpose(0, 2, 1))[bgrpD1[i, 0], :, :],
                                                                (Dgk[bgrpD1[i, 1]].transpose(0, 2, 1))[bgrpD1[i, 0], :, :]]) \
                                                      @ np.block([[TDgk[bgrp[i,1]][bgrpD1[i, 0], :, :]],
                                                                  [-1 * BB[bgrpD1[i, 1]][bgrpD1[i, 0], :, :]]]))
        if len(bgrpD2) != 0:
            for i in range(0, len(bgrpD2)):
                sD2[bgrpD2[i, 0], bgrpD2[i, 0], :, :] += -1*HB_inv[bgrpD2[i,0],: :] @ (np.block([(RB[bgrpD2[i, 1]].transpose(0, 2, 1))[bgrpD2[i, 0], :, :],
                                                                (Dgk[bgrpD2[i, 1]].transpose(0, 2, 1))[bgrpD2[i, 0], :, :]]) \
                                                      @ np.block([[TDgk[bgrp[i,1]][bgrpD2[i, 0], :, :]],
                                                                  [-1 * BB[bgrpD2[i, 1]][bgrpD2[i, 0], :, :]]]))
        if len(bgrpD3) != 0:
            for i in range(0, len(bgrpD3)):
                sD3[bgrpD3[i, 0], bgrpD3[i, 0], :, :] += -1*HB_inv[bgrpD3[i,0],: :] @ (np.block([(RB[bgrpD3[i, 1]].transpose(0, 2, 1))[bgrpD3[i, 0], :, :],
                                                                    (Dgk[bgrpD3[i, 1]].transpose(0, 2, 1))[bgrpD3[i, 0], :, :]]) \
                                                          @ np.block([[TDgk[bgrp[i,1]][bgrpD3[i, 0], :, :]],
                                                                      [-1 * BB[bgrpD3[i, 1]][bgrpD3[i, 0], :, :]]]))

        # add Neumann boundary SATs (and construct sN matrix to obtain the Neumann SAT contribution that go to the RHS)
        sN1 = (np.block(np.zeros((nelem * nnodes, nelem * nfp)))).reshape(nelem, nelem, nnodes, nfp)
        sN2 = (np.block(np.zeros((nelem * nnodes, nelem * nfp)))).reshape(nelem, nelem, nnodes, nfp)
        sN3 = (np.block(np.zeros((nelem * nnodes, nelem * nfp)))).reshape(nelem, nelem, nnodes, nfp)
        for i in range(0, len(bgrpN)):
            sI[bgrpN[i, 0], bgrpN[i, 0], :, :] += HB_inv[bgrpN[i,0],: :] @ ((RB[bgrpN[i, 1]].transpose(0,2,1))[bgrpN[i, 0], :, :] \
                                                  @ BB[bgrpN[i, 1]][bgrpN[i, 0], :, :] \
                                                  @ Dgk[bgrpN[i, 1]][bgrpN[i, 0], :, :])

        for i in range(0, len(bgrpN1)):
            if len(bgrpN1) != 0:
                sN1[bgrpN1[i, 0], bgrpN1[i, 0], :, :] += -1*HB_inv[bgrpN1[i,0],: :] @ ((RB[bgrpN1[i, 1]].transpose(0, 2, 1))[bgrpN1[i, 0], :, :] \
                                                      @ BB[bgrpN1[i, 1]][bgrpN1[i, 0], :, :])
        for i in range(0, len(bgrpN2)):
            if len(bgrpN2) != 0:
                sN2[bgrpN2[i, 0], bgrpN2[i, 0], :, :] += -1*HB_inv[bgrpN2[i,0],: :] @ ((RB[bgrpN2[i, 1]].transpose(0, 2, 1))[bgrpN2[i, 0], :, :] \
                                                      @ BB[bgrpN2[i, 1]][bgrpN2[i, 0], :, :])
        for i in range(0, len(bgrpN3)):
            if len(bgrpN3) != 0:
                sN3[bgrpN3[i, 0], bgrpN3[i, 0], :, :] += -1*HB_inv[bgrpN3[i,0],: :] @ ((RB[bgrpN3[i, 1]].transpose(0, 2, 1))[bgrpN3[i, 0], :, :] \
                                                      @ BB[bgrpN3[i, 1]][bgrpN3[i, 0], :, :])

        # reshape the 4D array of the SATs into 2D
        sI_mat = (sI.transpose(0, 2, 1, 3)).reshape(nelem*nnodes, nelem*nnodes)
        sI_mat = sparse.csr_matrix(sI_mat)

        # construct the forcing terms that go to right hand side
        if uD is None:
            uD = np.zeros((nelem*nfp, 3))
        if uN is None:
            uN = np.zeros((nelem*nfp, 3))
        # reshape the sD and sN matrices and change them to sparse matrices
        sD1_mat = sparse.csr_matrix((sD1.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp))
        sD2_mat = sparse.csr_matrix((sD2.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp))
        sD3_mat = sparse.csr_matrix((sD3.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp))

        sN1_mat = sparse.csr_matrix((sN1.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp))
        sN2_mat = sparse.csr_matrix((sN2.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp))
        sN3_mat = sparse.csr_matrix((sN3.transpose(0, 2, 1, 3)).reshape(nelem * nnodes, nelem * nfp))

        # get the forcing term obtained as a result of implementing the boundary SAT (goes to the RHS)
        fB = sD1_mat @ uD[:, 0] + sD2_mat @ uD[:, 1] + sD3_mat @ uD[:, 2] + sN1_mat @ uN[:, 0] + sN2_mat @ uN[:, 1] \
             + sN3_mat @ uN[:, 2]

        return sI_mat, fB