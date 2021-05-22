import numpy as np
from scipy import sparse
from src.sats import SATs


class SATsUnsteady:

    @staticmethod
    def advection_sat_2d(nnodes, nelem, HB, BB, RB, nxB, nyB, etoe, etof, bgrpD, a=np.array([1, 1]), sat_type='upwind',
                         domain_type='notperiodic'):

        nfp = int(nxB[0][0].shape[0])  # number of nodes per facet, also nfp = p+1
        dim = 2
        nface = dim + 1

        # set advection speed in the normal direction
        anB = [a + b for a, b in zip([i * a[0, 0] for i in nxB], [i * a[1, 0] for i in nyB])]

        # set flux type
        alpha = 1           # set upwind SAT by default
        if (sat_type.lower() == 'central' or sat_type.lower() =='symmetric'):
            alpha = 0       # set symmetric SAT

        # get the inverse of the norm matrix
        HB_inv = np.linalg.inv(HB)

        # construct the SAT matrix
        sI = sparse.lil_matrix((nnodes * nelem, nnodes * nelem), dtype=np.float64)

        for elem in range(0, nelem):
            for face in range(0, nface):
                if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpD):
                    sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += HB_inv[elem] \
                                                            @ (RB[face][elem].T @ BB[face][elem]
                                                               @ (1/2 * np.diag(anB[face][elem].flatten()) @ RB[face][elem]
                                                                  - alpha/2 * np.abs(np.diag(anB[face][elem].flatten())) @ RB[face][elem]))

                # SAT terms from neighboring elements -- i.e., the subtracted part in terms containing (uk - uv)
                elem_nbr = etoe[elem, face]
                if elem_nbr != elem:
                    nbr_face = etof[elem, face]
                    sI[elem*nnodes:(elem+1)*nnodes, elem_nbr*nnodes:(elem_nbr+1)*nnodes] += HB_inv[elem]\
                                            @ (RB[face][elem].T @ BB[face][elem]
                                               @ (-1/2 * np.diag(anB[face][elem].flatten()) @ np.flipud(RB[nbr_face][elem_nbr])
                                                  + alpha/2 * np.abs(np.diag(anB[face][elem].flatten())) @ np.flipud(RB[nbr_face][elem_nbr])))

        # construct SAT matrix that multiplies the Dirichlet boundary vector
        sD = sparse.lil_matrix((nelem * nnodes, nelem * nfp * nface), dtype=np.float64)
        # calculate SAT at boundaries
        if domain_type.lower() == 'notperiodic':
            for i in range(0, len(bgrpD)):
                elem = bgrpD[i, 0]
                face = bgrpD[i, 1]
                sI[elem*nnodes:(elem+1)*nnodes, elem*nnodes:(elem+1)*nnodes] += HB_inv[elem] \
                                                            @ (RB[face][elem].T @ BB[face][elem]
                                                               @ (np.diag(anB[face][elem].flatten()) @ RB[face][elem]))

                sD[elem*nnodes:(elem+1)*nnodes, (elem*nface*nfp+nfp*face):(elem*nface*nfp+nfp*(face+1))] += HB_inv[elem] \
                                                                                        @ (RB[face][elem].T @ BB[face][elem]
                                                                                           @ (np.diag(anB[face][elem].flatten())))

        return sI, sD

    @ staticmethod
    def burgers_sat_2d(u, uD, uN, nelem, HB, DxB, DyB, BB, RB, nxB, nyB, jacB, etoe, etof, bgrpD, bgrpN,
                       LxxB=1.0, LxyB=0.0, LyxB=0.0, LyyB=1.0, sat_inviscid='splitform', sat_viscous='BR2',
                       domain_type='notperiodic', nu=None):

        nfp = int(nxB[0][0].shape[0])  # number of nodes per facet, also nfp = p+1
        dim = 2
        nface = dim + 1
        nnodes = u.shape[0]

        # set advection speed in the normal direction
        anB = [a + b for a, b in zip([i * 1 for i in nxB], [i * 1 for i in nyB])]

        # get the inverse of the norm matrix
        HB_inv = np.linalg.inv(HB)

        # construct the SAT matrix
        u_sat_inv_twopoint1 = np.zeros(u.shape)
        u_sat_inv_twopoint2 = np.zeros(u.shape)
        u_sat_inv_twopoint3 = np.zeros(u.shape)
        u_sat_inv_splitform = np.zeros(u.shape)

        if sat_inviscid.lower() == 'splitform':
            for elem in range(0, nelem):
                for face in range(0, nface):
                    if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpD):
                        elem_nbr = etoe[elem, face]
                        nbr_face = etof[elem, face]

                        u_sat_inv_splitform[:, elem] += HB_inv[elem] @ (RB[face][elem].T @ BB[face][elem]
                                                             @ np.diag(anB[face][elem].flatten())
                                                         @ (1/3 * RB[face][elem] @ (u[:, elem]**2)\
                                                            - 1/6 * ((RB[face][elem] @ u[:, elem])\
                                                                     * (np.flipud(RB[nbr_face][elem_nbr]) @ u[:, elem_nbr])) \
                                                            - 1/6 * ((np.flipud(RB[nbr_face][elem_nbr]) @ u[:, elem_nbr])\
                                                                     * (np.flipud(RB[nbr_face][elem_nbr]) @ u[:, elem_nbr]))))

            if domain_type.lower() == 'notperiodic':
                for i in range(0, len(bgrpD)):
                    elem = bgrpD[i, 0]
                    face = bgrpD[i, 1]

                    u_sat_inv_splitform[:, elem] += HB_inv[elem] @ (RB[face][elem].T @ BB[face][elem] \
                                                     @ np.diag((anB[face][elem].flatten()))\
                                                     @ ((1 / 3 * (RB[face][elem] @ (u[:, elem]**2)))\
                                                        - 1 / 6 * ((RB[face][elem] @ u[:, elem]) * (uD[face*nfp:(face+1)*nfp, elem]))\
                                                        - 1 / 6 * (uD[face*nfp:(face+1)*nfp, elem] * uD[face*nfp:(face+1)*nfp, elem])))

            u_sat_inv = u_sat_inv_splitform

        elif sat_inviscid.lower() in ['twopoint', 'twopoint1']:
            for elem in range(0, nelem):
                for face in range(0, nface):
                    if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpD):
                        elem_nbr = etoe[elem, face]
                        nbr_face = etof[elem, face]

                        u_sat_inv_twopoint1[:, elem] += 1/6 * HB_inv[elem] @ (np.diag((u[:, elem].reshape((-1, 1))).flatten())
                                                                     @ RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (RB[face][elem] @ u[:, elem]
                                                                        - np.flipud(RB[nbr_face][elem_nbr]) @ u[:, elem_nbr])
                                                                     + RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (RB[face][elem] @ u[:, elem]**2
                                                                        - np.flipud(RB[nbr_face][elem_nbr]) @ u[:,elem_nbr]**2))

            if domain_type.lower() == 'notperiodic':
                for i in range(0, len(bgrpD)):
                    elem = bgrpD[i, 0]
                    face = bgrpD[i, 1]

                    u_sat_inv_twopoint1[:, elem] += 1/6 * HB_inv[elem] @ (np.diag((u[:, elem].reshape((-1, 1))).flatten())
                                                                     @ RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (RB[face][elem] @ u[:, elem]
                                                                        - uD[face*nfp:(face+1)*nfp, elem])
                                                                     + RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (RB[face][elem] @ u[:, elem]**2
                                                                        - uD[face*nfp:(face+1)*nfp, elem]**2))

                                                                     # - np.diag((u[:, elem].reshape((-1, 1))).flatten())**2
                                                                     # @ RB[face][elem].T @ BB[face][elem]
                                                                     # @ np.diag(anB[face][elem].flatten())
                                                                     # @ RB[face][elem] @ np.ones((nnodes, 1)).flatten())

                                                    # - (1/2 * HB_inv[elem] @ (((RB[face][elem].T @ BB[face][elem]
                                                    #                  @ np.diag(anB[face][elem].flatten())
                                                    #                  @ RB[face][elem]) * SATsUnsteady.burgers_two_point_flux(u[:, elem], u[:, elem]))
                                                    #                         @ np.ones((nnodes, 1)))).flatten()
            u_sat_inv = u_sat_inv_twopoint1

        elif sat_inviscid.lower() == 'twopoint2':
            for elem in range(0, nelem):
                for face in range(0, nface):
                    if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpD):
                        elem_nbr = etoe[elem, face]
                        nbr_face = etof[elem, face]

                        u_sat_inv_twopoint2[:, elem] += (1/2 * HB_inv[elem] @ (((RB[face][elem].T @ BB[face][elem]
                                                            @ np.diag(anB[face][elem].flatten()) @ RB[face][elem])
                                                            * SATsUnsteady.burgers_two_point_flux(u[:, elem], u[:, elem]))
                                                            @ np.ones((nnodes, 1)))).flatten()

                        u_sat_inv_twopoint2[:, elem] += -(1/2 * HB_inv[elem] @ (((RB[face][elem].T @ BB[face][elem]
                                                            @ np.diag(anB[face][elem].flatten())
                                                                      @ np.flipud(RB[nbr_face][elem_nbr]))
                                                            * SATsUnsteady.burgers_two_point_flux(u[:, elem], u[:, elem_nbr]))
                                                            @ np.ones((nnodes, 1)))).flatten()

            if domain_type.lower() == 'notperiodic':
                for i in range(0, len(bgrpD)):
                    elem = bgrpD[i, 0]
                    face = bgrpD[i, 1]

                    u_sat_inv_twopoint2[:, elem] += 1/6 * HB_inv[elem] @ (np.diag((u[:, elem].reshape((-1, 1))).flatten())
                                                                     @ RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (RB[face][elem] @ u[:, elem]
                                                                        - uD[face*nfp:(face+1)*nfp, elem])
                                                                     + RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (RB[face][elem] @ u[:, elem]**2
                                                                        - uD[face*nfp:(face+1)*nfp, elem]**2))

                                                                     # - np.diag((u[:, elem].reshape((-1, 1))).flatten())**2
                                                                     # @ RB[face][elem].T @ BB[face][elem]
                                                                     # @ np.diag(anB[face][elem].flatten())
                                                                     # @ RB[face][elem] @ np.ones((nnodes, 1)).flatten())

                    # u_sat_inv_twopoint2[:, elem] += -(1/2 * HB_inv[elem] @ ((RB[face][elem].T @ (BB[face][elem]
                    #                                                @ np.diag(anB[face][elem].flatten())
                    #                                               * SATsUnsteady.burgers_two_point_flux(RB[face][elem] @ u[:, elem],
                    #                                                                        uD[face*nfp:(face+1)*nfp, elem])))
                    #                                              @ np.ones((nfp, 1)))).flatten()

            u_sat_inv = u_sat_inv_twopoint2

        elif sat_inviscid.lower() == 'twopoint3':
            for elem in range(0, nelem):
                for face in range(0, nface):

                    u_sat_inv_twopoint3[:, elem] += (1/2 * HB_inv[elem] @ (((RB[face][elem].T @ BB[face][elem]
                                                            @ np.diag(anB[face][elem].flatten()) @ RB[face][elem])
                                                            * SATsUnsteady.burgers_two_point_flux(u[:, elem], u[:, elem]))
                                                            @ np.ones((nnodes, 1)))).flatten()

                    if not any(np.array_equal(np.array([elem, face]), rowB) for rowB in bgrpD):
                        elem_nbr = etoe[elem, face]
                        nbr_face = etof[elem, face]

                        u_sat_inv_twopoint3[:, elem] += -(1/2 * HB_inv[elem] @ (((RB[face][elem].T @ BB[face][elem]
                                                            @ np.diag(anB[face][elem].flatten())
                                                                      @ np.flipud(RB[nbr_face][elem_nbr]))
                                                            * SATsUnsteady.burgers_two_point_flux(u[:, elem], u[:, elem_nbr]))
                                                            @ np.ones((nnodes, 1)))).flatten()

            if domain_type.lower() == 'notperiodic':
                for i in range(0, len(bgrpD)):
                    elem = bgrpD[i, 0]
                    face = bgrpD[i, 1]

                    u_sat_inv_twopoint3[:, elem] += -1/6 * HB_inv[elem] @ (np.diag((u[:, elem].reshape((-1, 1))).flatten())
                                                                     @ RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (uD[face*nfp:(face+1)*nfp, elem])
                                                                     + RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ (uD[face*nfp:(face+1)*nfp, elem]**2)
                                                                     + np.diag((u[:, elem].reshape((-1, 1))).flatten())**2
                                                                     @ RB[face][elem].T @ BB[face][elem]
                                                                     @ np.diag(anB[face][elem].flatten())
                                                                     @ RB[face][elem] @ np.ones((nnodes, 1)).flatten())


            u_sat_inv = u_sat_inv_twopoint3

        # viscous SATs
        if nu != 0.0:
            sat = SATs.diffusion_sbp_sat_2d_steady(nnodes, nelem, LxxB, LxyB, LyxB, LyyB, DxB, DyB, HB, BB, nxB, RB, nyB, jacB,
                                             etoe, etof, bgrpD, bgrpN, flux_type=sat_viscous, uD=uD, uN=uN, eqn='primal')
            sI = sat['sI']
            fB = sat['fB']

            u_sat_vis = (sI @ u.reshape((-1, 1), order='F')).reshape((-1, nelem), order='F') + fB.reshape((-1, nelem), order='F')
            u_sat = u_sat_inv - u_sat_vis
        else:
            u_sat = u_sat_inv

        return u_sat

    @staticmethod
    def burgers_two_point_flux(uk, uv):

        nk = len(uk)
        nv = len(uv)
        F = np.zeros((nk, nv))

        for i in range(nk):
            for j in range(nv):
                F[i, j] = 2*(1/6 * (uk[i]**2 + uk[i]*uv[j] + uv[j]**2))

        return F