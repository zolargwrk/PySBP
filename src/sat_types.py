import numpy as np
from collections import deque

class SATs:

    @staticmethod
    def diffusion_dg_sat_1d(var, var_type, tl, tr, vmapM, vmapP, nx, mapI, mapO, varin, varout, flux_type='BR1', du=None):
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
        elif flux_type == 'BR2':
            if var_type=='u':
                dvar[0:] = (var_projF[vmapM][:, 0] - var_projF[vmapP][:, 0]) / 2
                # set boundary sats
                dvar[mapI] = (var_proj[mapI] - varin) / 2
                dvar[mapO] = (var_proj[mapO] - varout) / 2
            elif var_type=='q':
                dvar[0:] = var_projF[vmapM][:, 0] - var_projF[vmapP][:, 0]
                # set boundary sats
                dvar[mapI] = var_proj[mapI] - varin
                dvar[mapO] = var_proj[mapO] - varout

                alpha_r = 1
                dvar = dvar/2 + alpha_r*nx_tempC*(du.reshape((nface * nelem, 1), order='F'))

        dvar = dvar.reshape((nface, nelem), order='F')
        return dvar

    @staticmethod
    def diffusion_sbp_sat_1d(u, d_mat, h_mat, tl, tr, vmapM, vmapP, nx, mapI, mapO, uin, uout, rx,
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
        h_inv = np.linalg.inv(h_mat)

        # SAT coefficients for different methods
        T1 = 0
        T2k = 0
        T2v = 0
        T3k = 0
        T3v = 0
        T4 = 0
        T5k = 0
        T5v = 0
        T6k = 0
        T6v = 0

        if flux_type == 'BR1':
            eta = 2
            T1 = eta*(1/4*(tr.T @ h_inv @ tr + tl.T @ h_inv @ tl))[0, 0]
            T2k = (-1/2)
            T3k = (1/2)
            T2v = (-1/2)
            T3v = (1/2)
            T5k = (-1/4*(tl.T @ h_inv @ tl))[0, 0]
            T5v = (-1/4*(tr.T @ h_inv @ tr))[0, 0]
            T6k = (1/4*(tr.T @ h_inv @ tr))[0, 0]
            T6v = (1/4*(tl.T @ h_inv @ tl))[0, 0]
        elif flux_type == 'BR2':
            eta = nface
            T1 = (eta/4 * (tr.T @ h_inv @ tr + tl.T @ h_inv @ tl))[0, 0]
            T2k = (-1/2)
            T3k = (1/2)
            T2v = (-1/2)
            T3v = (1/2)
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
            T1 = (tr.T @ h_inv @ tr)[0, 0]
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
            Dgk = nx[0, 1] * (tr.T @ db_mat)  # D_{\gamma k}
            Dgv = nx[0, 0] * (tl.T @ db_mat)  # D_{\gamma v}
        else:
            Dgk = nx[0, 1]*(tr.T @ np.diag(b) @ d_mat)   # D_{\gamma k}
            Dgv = nx[0, 0]*(tl.T @ np.diag(b) @ d_mat)   # D_{\gamma v}
        # Dgk = nx[0, 1] * (tr.T @ np.diag(b) @ d_mat)  # D_{\gamma k}
        # Dgv = nx[0, 0] * (tl.T @ np.diag(b) @ d_mat)
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
        Ck = np.array([[T1, T3k], [T2k, T4]])
        Cv = np.array([[T1, T3v], [T2v, T4]])
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

        sI = h_inv @ (sat_k + sat_v + sat_P + sat_M + sat_PP + sat_MM)

        # boundary SATs for non-periodic boundary condition
        sB = None
        if boundary_type != 'Periodic':
            TD_left = 1 / 2 * (tl.T @ h_inv @ tl)[0, 0]     # Dirichlet boundary flux coefficient at the left bc
            TD_right = 1 / 2 * (tr.T @ h_inv @ tr)[0, 0]    # Dirichlet boundary flux coefficient at the right bc
            CDk = np.array([[TD_right], [-1]])      # coefficient for dirichlet boundary at the right boundary
            CDv = np.array([[TD_left], [-1]])       # coefficient for dirichlet boundary at the left boundary
            DnNk = (Dgk @ u[:, -1]).flatten()       # derivative of u at right boundary (for Neumann bc implementation)
            DnNv = (Dgv @ u[:, 0]).flatten()        # derivative of u at left boundary (for Neumann bc implementation)
            WDk = np.block([[tr.reshape(n, 1), Dgk.reshape(n, 1)]])
            WDv = np.block([[tl.reshape(n, 1), Dgv.reshape(n, 1)]])

            sD_left = 0
            sD_right = 0
            sN_left = 0
            sN_right = 0
            if uD_left != None:
                UDv = u_projv[0] - uD_left  # difference in u and Dirichlet bc at the left boundary
                sD_left = (WDv @ CDv) * UDv
            if uD_right != None:
                UDk = u_projk[-1] - uD_right  # difference in u and Dirichlet bc at the right boundary
                sD_right = (WDk @ CDk) * UDk
            if uN_left != None:
                sN_left = tl.T*(DnNv - uN_left)
            if uN_right != None:
                sN_right = tr.T*(DnNk - uN_right)

            sB = sD_left + sD_right + sN_left + sN_right

        return sI, sB