import numpy as np


class SATs:

    @staticmethod
    def diffusion_sat_1d(u, tl, tr, vmapM, vmapP, nx,  mapI, mapO, vmapI, vmapO, uin, uout, flux_type='BR1'):
        # remember that you can use q (derivative of u) in place of u to get SATs on q
        nface = 2
        n = u.shape[0]
        nelem = u.shape[1]

        # project u to the interfaces to get the u-u* at the faces
        u_proj = u.copy()
        u_projM = (u.T @ tl)[:, 0]
        u_projP = (u.T @ tr)[:, 0]
        u_proj[0, :] = u_projM
        u_proj[n-1, :] = u_projP

        du = np.zeros((nface, nelem))
        du = du.reshape((nface * nelem, 1), order='F')

        # reshape some matrices and calculate du
        u_projF = u_proj.reshape((n * nelem, 1), order='F')
        nx_tempC = nx.reshape((nface * nelem, 1), order='C')
        u_proj = np.reshape(np.array([u_projM, u_projP]), (2*len(u_projM), 1), order='F')

        if flux_type == 'BR1':
            du[0:] = (u_projF[vmapM][:, 0] - u_projF[vmapP][:, 0]) / 2
            # set boundary sats
            du[mapI] = (u_proj[mapI]-uin)/2
            du[mapO] = (u_proj[mapO]-uout)/2

        du = du.reshape((nface, nelem), order='F')

        return du

