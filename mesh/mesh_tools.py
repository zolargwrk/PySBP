import numpy as np
from types import SimpleNamespace
from mesh.mesh_generator import MeshGenerator1D, MeshGenerator2D
from src.ref_elem import Ref2D_DG, Ref2D_SBP
from src.calc_tools import CalcTools
from scipy import sparse


class MeshTools1D:
    """Collects methods used for operations on the meshes
        Some methods are adopted from the book by Hesthaven, Jan S., and Tim Warburton.
        Nodal discontinuous Galerkin methods: algorithms, analysis, and applications.
        Springer Science & Business Media, 2007."""

    @staticmethod
    def normals_1d(nelem):
        nx = np.zeros((nelem, 2))
        nx[:, 0] = -1
        nx[:, 1] = 1
        return nx

    @staticmethod
    def connectivity_1d (etov):
        nelem = etov.shape[0]   # number of elments
        nv = nelem+1            # number of vertices
        nface = 2               # number of face per element
        tot_face = nface*nelem  # total number of faces
        vn = np.array([[0], [1]], np.int)    # local face to local vertex connection

        # face to vertex and face to face connectivities
        ftov = np.zeros((2*nelem, nelem+1), np.int)
        k = 0
        for elem in range(0, nelem):
            for face in range(0, nface):
                ftov[k, etov[elem, vn[face, 0]]] = 1
                k = k+1
        ftof = ftov @ (ftov.T) - np.eye(tot_face, dtype=int)

        # complete face to face connectivity
        (faces2, faces1) = ftof.nonzero()

        # convert global face number to element and face number
        elem1 = np.array([np.floor(faces1/nface)], dtype=int).T
        face1 = np.array([np.mod(faces1, nface)], dtype=int).T
        elem2 = np.array([np.floor(faces2/nface)], dtype=int).T
        face2 = np.array([np.mod(faces2, nface)], dtype=int).T

        # element to element and face to face connectivities
        ind = MeshTools1D.sub2ind((nelem, nface), elem1, face1)
        etoe = np.array([range(0, nelem)]).T @ np.ones((1, nface), dtype=int)
        etof = np.array([np.ones((nelem, 1)) @ np.array([range(0, nface)])], dtype=int)

        etoe = etoe.reshape((nelem*nface, 1), order='F')
        etoe[ind] = elem2
        etoe = etoe.reshape((nelem, nface), order='F')

        etof = etof.reshape((nelem * nface, 1), order='F')
        etof[ind] = face2
        etof = etof .reshape((nelem, nface), order='F')

        return {'etoe': etoe, 'etof': etof}

    @staticmethod
    def sub2ind(array_shape, v1, v2):
        """returns the linear index equivalents to the row (v1) and column (v2) subscripts
        Inputs: array_shape - the shape of the array
                v1 - the row subscript
                v2 - the column subscript
        Output: ind - index of the linear array
        e.g., if the a = [[a01, a02], [a11, a11]]
            sub2ind((2,2), 1, 0) = 2
            because the linear array is: a = [a01, a02, a10, a11] and the [0,1] subscript is now located at a[2]"""
        ind = np.asarray(v1 + v2*(array_shape[0]), dtype=int)
        return ind.T

    @staticmethod
    def buildmaps_1d(n, x, a, etoe, etof, fmask, boundary_type=None):
        # n : number of nodes per element, -->  n = p + 1 for LG and LGL operators
        nelem = etoe.shape[0]
        nface = 2
        nodeids = np.reshape([np.arange(0, nelem*n)], (nelem, n)).T
        vmapM = np.zeros((1, nface, nelem), dtype=int)
        vmapP = np.zeros((1, nface, nelem), dtype=int)

        for i in range(0, nelem):
            for j in range(0, nface):
                vmapM[:, j, i] = int(nodeids[fmask[j], i])

        x = x.reshape((n, nelem), order='F')

        for i in range(0, nelem):
            for j in range(0, nface):
                i2 = etoe[i, j]
                j2 = etof[i, j]
                vidM = vmapM[:, j, i]
                vidP = vmapM[:, j2, i2]
                vmapP[:, j, i] = vidP

        vmapP = vmapP.reshape((nelem*nface, 1), order='F')
        vmapM = vmapM.reshape((nelem*nface, 1), order='F')

        # boundary data
        mapB = np.asarray(vmapP == vmapM).nonzero()
        vmapB = vmapM[mapB]

        # maps at inflow and outflow
        if a >= 0:
            mapI = 0
            mapO = nelem*nface - 1
            vmapI = 0
            vmapO = nelem*n - 1
        else:
            mapI = nelem * nface - 1
            mapO = 0
            vmapI = nelem * n - 1
            vmapO = 0

        if boundary_type == 'Periodic':
            vmapP[0] = vmapM[-1]
            vmapP[-1] = vmapM[0]

        return {'vmapM': vmapM, 'vmapP': vmapP, 'vmapB': vmapB, 'mapB': mapB, 'mapI': mapI, 'mapO': mapO,
                'vmapI': vmapI, 'vmapO': vmapO}

    @staticmethod
    def geometric_factors_1d(x, d_mat_ref):
        """Computes the 1D mesh Jacobian
        inputs: x - nodal location of an element on the physical domain
                d_mat_ref - derivative operator on the reference element
        outputs: rx - the derivative of the reference x with respect to the physical x
                 jac - the transformation Jacobian"""
        n = d_mat_ref.shape[0]
        nelem = int(len(x)/n)
        x = x.reshape((n, nelem), order='F')
        jac = d_mat_ref @ x
        rx = 1/jac
        return {'rx': rx, 'jac': jac}

    @staticmethod
    def hrefine_uniform_1d(rhs_data):
        # unpack data
        rdata = SimpleNamespace(**rhs_data)
        fx = rdata.fx   # the facet vertex coordinate value of the reference elemment
        x = rdata.x     # physical coordinate
        x_ref = rdata.x_ref
        n = rdata.n     # number of degrees of freedom on the reference element

        # find the center of the elements (this will be changed with marked elements for adaptive mesh refinement)
        xc = np.mean([fx[0, :], fx[1, :]], axis=0)

        # add vertices at the center of the elements
        velem = np.sort(np.hstack([fx.reshape((1, np.prod(fx.shape))).flatten(), xc.flatten()]))
        velem = np.unique(np.floor(1e12*velem)/1e12)    # identify unique vertices

        # renumber the vertices
        nv = len(velem)   # update the total number of vertices
        nelem = nv - 1    # update the total number of elements
        xl_elem = velem[0:-1]
        xr_elem = velem[1:]

        # affine mapping to the physical elements
        coord_elem = np.zeros((1, nelem, n))
        for elem in range(0, nelem):
            coord_elem[0, elem, :] = 1 / 2 * (xl_elem[elem] * (1 - x_ref) + xr_elem[elem] * (1 + x_ref))

        coord_elem = coord_elem[0, :, :].T

        # identify vertex to element connectivity
        convty = np.zeros((nelem, 2))
        convty[:, 0] = xl_elem
        convty[:, 1] = xr_elem

        convty_id = np.array([range(0, nelem), range(1, nelem + 1)], np.int).T

        # element to vertex connectivity
        etov = convty_id

        # boundary group
        bgrp = np.zeros((2, 1), np.int)
        bgrp[0] = 0
        bgrp[1] = nelem*n - 1

        return {'x': x, 'etov': etov, 'x_ref': x_ref, 'bgrp': bgrp, 'coord_elem': coord_elem, 'nelem': nelem}

    @staticmethod
    def trad_refine_uniform_1d(rhs_data, p, quad_type, var_coef=None, app=1):
        rdata = SimpleNamespace(**rhs_data)
        x = rdata.x     # physical coordinate
        n = 2*rdata.n     # number of degrees of freedom on the reference element
        xl = x[0,0]
        xr = x[-1,0]
        nelem = rdata.nelem
        if var_coef == None:
            b = np.ones((1, n))
        else:
            b = var_coef(n)

        mesh_info = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        mesh = SimpleNamespace(**mesh_info)

        return {'x': mesh.x, 'etov': mesh.etov, 'x_ref': mesh.x_ref, 'bgrp': mesh.bgrp,
                'coord_elem': mesh.coord_elem, 'nelem': mesh.nelem, 'n': n}


class MeshTools2D:

    @staticmethod
    def geometric_factors_2d(x, y, Dr, Ds):
        xr = Dr @ x
        xs = Ds @ x
        yr = Dr @ y
        ys = Ds @ y
        jac = xr*ys - xs*yr
        rx = ys/jac
        sx = -yr/jac
        ry = -xs/jac
        sy = xr/jac

        return {'xr': xr, 'xs': xs, 'yr': yr, 'ys': ys, 'jac': jac, 'rx': rx, 'sx': sx, 'ry': ry, 'sy': sy}

    @staticmethod
    def facet_geometric_factors_2d(xf, yf, Drf, Dsf):
        nfp = int(xf.shape[0]/3)
        fid1 = np.arange(0, nfp)
        fid2 = np.arange(nfp, 2*nfp)
        fid3 = np.arange(2*nfp, 3*nfp)

        xrf1 = Drf[0] @ xf[fid1]
        yrf1 = Drf[0] @ yf[fid1]

        xrf2 = Drf[1] @ xf[fid2]
        xsf2 = Drf[1] @ xf[fid2]
        yrf2 = Drf[1] @ yf[fid2]
        ysf2 = Drf[1] @ yf[fid2]

        xsf3 = Drf[0] @ xf[fid3]
        ysf3 = Drf[0] @ yf[fid3]

        surf_jacf1 = np.sqrt((xrf1**2 + yrf1**2))
        surf_jacf2 = np.sqrt((xrf2+xsf2)**2 + (yrf2+ysf2)**2)/np.sqrt(2)
        surf_jacf3 = np.sqrt((xsf3**2 + ysf3**2))

        return {'xrf1': xrf1, 'yrf1': yrf1, 'xrf2': xrf2, 'xsf2': xsf2, 'yrf2': yrf2, 'ysf2': ysf2, 'xsf3': xsf3,
                'ysf3': ysf3, 'surf_jacf1': surf_jacf1, 'surf_jacf2': surf_jacf2, 'surf_jacf3': surf_jacf3}

    @staticmethod
    def connectivity_2d(etov):
        # number of faces, elements, and vertices
        nface = 3
        nelem = etov.shape[0]
        nvert = np.max(np.max(etov))+1

        # create list of faces 1, 2 and 3
        fnodes = np.array([etov[:, [0, 1]], etov[:, [1, 2]], etov[:, [2, 0]]])
        fnodes = np.sort(fnodes.reshape(3*nelem, 2), 1)
        fnodes = np.sort(fnodes[:], 1)

        # default element to element and element to face connectivity
        etoe = np.arange(0, nelem).reshape((nelem, 1)) @ np.ones((1, nface))
        etoe = etoe.astype(int)
        etof = np.ones((nelem, 1)) @ np.arange(0, nface).reshape((1, nface))
        etof = etof.astype(int)

        # give unique id number for faces using their node number
        id = fnodes[:, 0]*nvert + fnodes[:, 1] + 1
        vtov = np.asarray([id.reshape(nelem*nface, 1), np.array((np.arange(0, nelem*nface)).reshape((nelem*nface, 1), order='F')),
                                np.array(etoe.reshape((nelem*nface, 1), order='F')), np.array(etof.reshape((nface*nelem, 1), order='F'))])
        vtov = (vtov.reshape((nelem*nface*4, 1))).reshape((nelem*nface, 4), order='F')

        # # give unique id number for faces using their node number
        # id = fnodes[:, 0] * nvert + fnodes[:, 1] + 1

        # sort by global face number (first column)
        sorted = vtov[vtov[:, 0].argsort(), ]

        # find matches
        indx = np.where(sorted[0:-1, 0] == sorted[1:, 0])[0]

        # match
        matchL = np.vstack([sorted[indx, :], sorted[indx + 1, :]])
        matchR = np.vstack([sorted[indx + 1, :], sorted[indx, :]])

        etoe_temp = (etoe.reshape((etoe.shape[0]*etoe.shape[1], 1), order='F'))
        etoe_temp[matchL[:, 1].T] = np.asarray(matchR[:, 2]).reshape(len(matchR[:, 2]), 1)
        etoe = etoe_temp.reshape((nelem, 3), order='F')

        etof_temp = (etof.reshape((etof.shape[0]*etof.shape[1], 1), order='F'))
        etof_temp[matchL[:, 1].T] = np.asarray(matchR[:, 3]).reshape(len(matchR[:, 3]), 1)
        etof = etof_temp.reshape((nelem, 3), order='F')

        return {'etoe': etoe, 'etof': etof}

    @staticmethod
    def connectivity_sbp_2d(etov):
        """Returns the element to element and element to facet connectivity where the reference element has facet
        numbering such that the facet numbers are the same as the vertex number opposite to them (i.e., the vertex
        that is not on at the ends of the facets)"""
        # number of faces, elements, and vertices
        nface = 3
        nelem = etov.shape[0]
        nvert = np.max(np.max(etov))+1

        # create list of faces 1, 2 and 3
        # fnodes = np.array([etov[:, [1, 2]], etov[:, [2, 0]], etov[:, [0, 1]]])
        fnodes = np.array([etov[:, [0, 1]], etov[:, [1, 2]], etov[:, [2, 0]]])
        fnodes = np.sort(fnodes.reshape(3*nelem, 2), 1)
        fnodes = np.sort(fnodes[:], 1)

        # default element to element and element to face connectivity
        etoe = np.arange(0, nelem).reshape((nelem, 1)) @ np.ones((1, nface))
        etoe = etoe.astype(int)
        etof = np.ones((nelem, 1)) @ np.arange(0, nface).reshape((1, nface))
        etof = etof.astype(int)

        # give unique id number for faces using their node number
        id = fnodes[:, 0]*nvert + fnodes[:, 1] + 1
        vtov = np.asarray([id.reshape(nelem*nface, 1), np.array((np.arange(0, nelem*nface)).reshape((nelem*nface, 1), order='F')),
                                np.array(etoe.reshape((nelem*nface, 1), order='F')), np.array(etof.reshape((nface*nelem, 1), order='F'))])
        vtov = (vtov.reshape((nelem*nface*4, 1))).reshape((nelem*nface, 4), order='F')

        # # give unique id number for faces using their node number
        # id = fnodes[:, 0] * nvert + fnodes[:, 1] + 1

        # sort by global face number (first column)
        sorted = vtov[vtov[:, 0].argsort(), ]

        # find matches
        indx = np.where(sorted[0:-1, 0] == sorted[1:, 0])[0]

        # match
        matchL = np.vstack([sorted[indx, :], sorted[indx + 1, :]])
        matchR = np.vstack([sorted[indx + 1, :], sorted[indx, :]])
        matchL = matchL.astype(int)
        matchR = matchR.astype(int)

        etoe_temp = (etoe.reshape((etoe.shape[0]*etoe.shape[1], 1), order='F'))
        etoe_temp[matchL[:, 1].T] = np.asarray(matchR[:, 2]).reshape(len(matchR[:, 2]), 1)
        etoe = etoe_temp.reshape((nelem, 3), order='F')

        etof_temp = (etof.reshape((etof.shape[0]*etof.shape[1], 1), order='F'))
        etof_temp[matchL[:, 1].T] = np.asarray(matchR[:, 3]).reshape(len(matchR[:, 3]), 1)
        etof = etof_temp.reshape((nelem, 3), order='F')

        # etoe = np.roll(etoe, -1, axis=1)
        # etof = np.roll(etof, -1, axis=1)

        return {'etoe': etoe, 'etof': etof}

    @staticmethod
    def affine_map_2d(vx, vy, r, s, etov):
        va = etov[:, 0].T
        vb = etov[:, 1].T
        vc = etov[:, 2].T

        x = 0.5*(-(r+s)*(vx[va]).flatten() + (1+r)*(vx[vb]).flatten() + (1+s)*(vx[vc]).flatten())
        y = 0.5*(-(r+s)*(vy[va]).flatten() + (1+r)*(vy[vb]).flatten() + (1+s)*(vy[vc]).flatten())

        # something similar but for volume nodes (using barycentric coordinates, bary is barycentric for the volume nodes)
        # if bary is not None:
        #     vert1 = np.array([[vx[va], vy[va]], [vx[vb], vy[vb]], [vx[vc], vy[vc]]])
        #     vert2 = vert1.transpose(0,2,1).reshape(3, 2*len(va))
        #     xy = Ref2D_SBP.barycentric_to_cartesian(bary, vert2)
        #     x = xy[:, 0::2]
        #     y = xy[:, 1::2]

        return x, y

    @staticmethod
    def affine_map_facet_sbp_2d(vx, vy, etov, baryf):
        # vertices
        va = etov[:, 0].T
        vb = etov[:, 1].T
        vc = etov[:, 2].T

        # we can also use barycentric coordinates as follows (for nodes at the facet that do not coincide with volume nodes)
        # get vertices on each facets
        vert1 = np.array([[vx[va], vy[va]], [vx[vb], vy[vb]]])
        v1 = vert1.transpose(0, 2, 1).reshape(2, 2 * len(va))
        vert2 = np.array([[vx[vb], vy[vb]], [vx[vc], vy[vc]]])
        v2 = vert2.transpose(0, 2, 1).reshape(2, 2 * len(va))
        vert3 = np.array([[vx[vc], vy[vc]], [vx[va], vy[va]]])
        v3 = vert3.transpose(0, 2, 1).reshape(2, 2 * len(va))

        # calculate the coordinates of nodes on the facets
        xy1 = Ref2D_SBP.barycentric_to_cartesian(baryf, v1)
        xy2 = Ref2D_SBP.barycentric_to_cartesian(baryf, v2)
        xy3 = Ref2D_SBP.barycentric_to_cartesian(baryf, v3)

        # get x and y coordinates of nodes on the facet
        xy = np.vstack([xy1, xy2, xy3])
        xf = xy[:, 0::2]
        yf = xy[:, 1::2]

        return xf, yf

    @staticmethod
    def buildmaps_2d(p, n, x, y, etov, etoe, etof, fmask, boundary_type=None):
        # n = (p+1)*(p+2)/2 : number of degrees of freedom per element
        nelem = etov.shape[0]
        nfp = p+1   # number of nodes on each facet
        nface = 3
        nodeids = (np.arange(0, n*nelem)).reshape((n, nelem), order='F')
        vmapM = np.zeros((nelem, nface, nfp), dtype=int)
        vmapP = np.zeros((nelem, nface,  nfp), dtype=int)
        mapM = (np.arange(0, nelem*nfp*nface)).reshape((nelem*nfp*nface, 1))
        mapP = (mapM.copy()).reshape((nelem, nface, nfp))

        for elem in range(0, nelem):
            for face in range(0, nface):
                vmapM[elem, face, :] = nodeids[fmask[:, face], elem]

        for elem in range(0, nelem):
            for face in range(0, nface):
                elem2 = etoe[elem, face]    # neighboring element
                face2 = etof[elem, face]    # neighboring face

                # reference length
                x_vec = x.reshape((nelem*n, 1), order='F')
                y_vec = y.reshape((nelem*n, 1), order='F')

                # volume node number on left and right element
                vidM = vmapM[elem, face, :]
                vidP = vmapM[elem2, face2, :]

                # obtain the coordinate values of the facet nodes on the left and right element
                x1 = x_vec[vidM] @ np.ones((1, nfp), dtype=int)
                y1 = y_vec[vidM] @ np.ones((1, nfp), dtype=int)
                x2 = x_vec[vidP] @ np.ones((1, nfp), dtype=int)
                y2 = y_vec[vidP] @ np.ones((1, nfp), dtype=int)

                # calulate the distance between each nodes on the neighboring elements (distance matrix)
                distance = (x1 - x2.T)**2 + (y1 - y2.T)**2

                # find nodes sharing a coordinate (distance = 0)
                (idP, idM) = np.where(np.sqrt(distance) < 1e-12)
                # find the vertex numbers on the right element (vmapP)
                vmapP[elem, face, idM] = vidP[idP]

                # global numbering to nodes on the facet (relative to mapM which number the nodes from 0 to nfp*nface*nelem -1)
                mapP[elem, face, idM] = idP + face2*nfp + elem2*nface*nfp

        if boundary_type == 'Periodic':
            vmapP[0] = vmapM[-1]
            vmapP[-1] = vmapM[0]
            mapP[0] = mapM[-1]
            mapP[-1] = mapM[0]

        # reshape the maps into vectors
        vmapM = ((vmapM.copy()).transpose(0, 1, 2)).reshape(nelem*nfp*nface, 1)
        vmapP = ((vmapP.copy()).transpose(0, 1, 2)).reshape(nelem*nfp*nface, 1)
        mapP = ((mapP.copy()).transpose(0, 1, 2)).reshape(nelem*nfp*nface, 1)

        # obtain list of boundary nodes
        mapB = np.where(vmapM == vmapP)[0]
        mapB = mapB.reshape((len(mapB), 1))
        vmapB = (vmapM[mapB]).reshape(len(mapB), 1)

        return {'mapM': mapM, 'mapP': mapP, 'vmapM': vmapM, 'vmapP': vmapP, 'vmapB': vmapB, 'mapB': mapB}

    @staticmethod
    def normals_2d(p, x, y, Dr, Ds, fmask):
        nface = 3
        nfp = p+1
        nelem = x.shape[1]
        # obtain the Jacobian
        xr = Dr @ x
        yr = Dr @ y
        xs = Ds @ x
        ys = Ds @ y
        jac = xr*ys - xs*yr

        # obtain geometric factors at the face nodes
        fxr = xr[fmask.flatten(order='F'), :]
        fxs = xs[fmask.flatten(order='F'), :]
        fyr = yr[fmask.flatten(order='F'), :]
        fys = ys[fmask.flatten(order='F'), :]

        # build normals and face ids
        nx = np.zeros((nface*nfp, nelem))
        ny = np.zeros((nface*nfp, nelem))
        fid1 = (np.arange(0, nfp)).reshape((nfp, 1))
        fid2 = (np.arange(nfp, 2*nfp)).reshape((nfp, 1))
        fid3 = (np.arange(2*nfp, 3*nfp)).reshape((nfp, 1))

        # # The normals are computed as shown in Fig. 6.1 of the Nodal DG book by Hesthaven and following his code
        # face 0
        nx[fid1, :] = fyr[fid1, :]
        ny[fid1, :] = -fxr[fid1, :]
        # face 1
        nx[fid2, :] = fys[fid2, :] - fyr[fid2, :]
        ny[fid2, :] = -fxs[fid2, :] + fxr[fid2, :]
        # face 3
        nx[fid3, :] = -fys[fid3, :]
        ny[fid3, :] = fxs[fid3, :]

        # # face 3
        # nx[fid3, :] = fyr[fid1, :]
        # ny[fid3, :] = -fxr[fid1, :]
        # # face 1
        # nx[fid1, :] = (fys[fid2, :] - fyr[fid2, :])
        # ny[fid1, :] = (-fxs[fid2, :] + fxr[fid2, :])
        # # face 3
        # nx[fid2, :] = -fys[fid3, :]
        # ny[fid2, :] = fxs[fid3, :]

        # normalize
        surf_jac = np.sqrt(nx*nx + ny*ny)
        nx = nx/surf_jac
        ny = ny/surf_jac

        return {'nx': nx, 'ny': ny, 'surf_jac': surf_jac}

    @staticmethod
    def normals_sbp_2d(rx, ry, sx, sy, jac, R1, R2, R3):
        """Calculates the surface normals of the physical element given the vertices."""
        # get number of nodes per face and number of elments
        nfp = R1.shape[0]

        # calculate the geometric factors at each facet
        rxf1 = R1 @ rx
        sxf1 = R1 @ sx
        rxf2 = R2 @ rx
        sxf2 = R2 @ sx
        rxf3 = R3 @ rx
        sxf3 = R3 @ sx

        ryf1 = R1 @ ry
        syf1 = R1 @ sy
        ryf2 = R2 @ ry
        syf2 = R2 @ sy
        ryf3 = R3 @ ry
        syf3 = R3 @ sy

        jacf1 = R1 @ jac
        jacf2 = R2 @ jac
        jacf3 = R3 @ jac

        # calculate normals at the facets
        # # jac_all_face = np.repeat(jac[0, :], nfp).reshape((nfp, -1), order="F")
        # nx1 = (rxf1 + sxf1)*jac_all_face
        # ny1 = (ryf1 + syf1)*jac_all_face
        #
        # nx2 = -rxf2*jac_all_face
        # ny2 = -ryf2*jac_all_face
        #
        # nx3 = -sxf3*jac_all_face
        # ny3 = -syf3*jac_all_face

        nx2 = (rxf2 + sxf2)*jacf2 / (np.sqrt(2))
        ny2 = (ryf2 + syf2)*jacf2 / (np.sqrt(2))

        nx3 = -rxf3*jacf3
        ny3 = -ryf3*jacf3

        nx1 = -sxf1*jacf1
        ny1 = -syf1*jacf1

        # get the normals into one matrix
        nx = np.vstack([nx1, nx2, nx3])
        ny = np.vstack([ny1, ny2, ny3])

        # get the magnitude of the surface jacobian
        # surf_jac_scaling = np.repeat(np.array([1/np.sqrt(2), 1, 1]), nfp).reshape((nfp*3, -1), order="F")
        surf_jac = np.sqrt(nx**2 + ny**2)
        nx = nx / surf_jac
        ny = ny / surf_jac

        return {'nx': nx, 'ny': ny, 'surf_jac': surf_jac}

    @staticmethod
    def normals_sbp_curved_2d(xrf, yrf, xsf, ysf):
        """Calculates the surface normals of the physical element given the vertices."""

        nfp = int(xrf.shape[0]/3)
        fid1 = np.arange(0, nfp)
        fid2 = np.arange(nfp, 2 * nfp)
        fid3 = np.arange(2 * nfp, 3 * nfp)

        # calculate the surface jacobian
        surf_jacf1 = np.sqrt((xrf[fid1, :] ** 2 + yrf[fid1, :] ** 2))
        surf_jacf2 = np.sqrt((-xrf[fid2, :] + xsf[fid2, :]) ** 2 + (-yrf[fid2, :] + ysf[fid2, :]) ** 2) / np.sqrt(2)
        surf_jacf3 = np.sqrt((xsf[fid3, :] ** 2 + ysf[fid3, :] ** 2))

        # calculate the normals at the faces
        nx1 = 1 / surf_jacf1 * (yrf[fid1, :])
        ny1 = 1 / surf_jacf1 * (-xrf[fid1, :])

        nx2 = 1 / np.sqrt(2) * 1 / surf_jacf2 * (ysf[fid2, :] - yrf[fid2, :])
        ny2 = 1 / np.sqrt(2) * 1 / surf_jacf2 * (-xsf[fid2, :] + xrf[fid2, :])

        nx3 = 1 / surf_jacf3 * (-ysf[fid3, :])
        ny3 = 1 / surf_jacf3 * (xsf[fid3, :])

        # get the normals into one matrix
        nx = np.vstack([nx1, nx2, nx3])
        ny = np.vstack([ny1, ny2, ny3])

        # get the surface jacobian in one matrix
        # surf_jac = np.vstack([surf_jacf1, surf_jacf2/(2*np.sqrt(2)), surf_jacf3])
        surf_jac = np.vstack([surf_jacf1, surf_jacf2, surf_jacf3])

        return {'nx': nx, 'ny': ny, 'surf_jac': surf_jac}

    @staticmethod
    def mesh_bgrp(nelem, bgrp, edge):
        """Includes element number and local face number to the boundary information contained in bgrp"""

        for ibgrp in range(0, len(bgrp)):
            # find indices where the boundary edge number matches those contained in edge
            s1 = np.char.array(edge[:, 0]*10) + np.char.array(edge[:, 1]*10)
            s2 = np.char.array(bgrp[ibgrp][:, 0]*10) + np.char.array(bgrp[ibgrp][:, 1]*10)
            ind_edge = np.where(np.in1d(s1, s2))[0]

            # get element number of elements containing the boundary edge
            # this is done using the fact that edge is obtained by reshaping element to vertex connectivity
            # see "mid_edge" method of MeshGenerator2D class in mesh_generator.py file
            belem = list()
            bface = np.zeros((len(ind_edge), 1), dtype=int)

            # if index is below nelem, the element number doesn't change and the local face number is 1 due to
            # how the edge matrix is constructed in "mid_edge" method
            belem.append(ind_edge[np.where(ind_edge < nelem)])
            bface[np.where(ind_edge < nelem)] = 1

            # if index is >= nelem but < 2*nelem, element number = ind_edge - nelem and local face number is 2
            belem.append(ind_edge[np.where(np.logical_and(nelem <= ind_edge, ind_edge < 2*nelem))] - nelem)
            bface[np.where(np.logical_and(nelem <= ind_edge, ind_edge < 2*nelem))] = 2

            # if index is >= 2*nelem  but < 3*nelem, element number = ind_edge - 2*nelem and local face number is 0
            belem.append(ind_edge[np.where(np.logical_and(2*nelem <= ind_edge, ind_edge < 3 * nelem))] - 2*nelem)
            bface[np.where(np.logical_and(2*nelem <= ind_edge, ind_edge < 3*nelem))] = 0

            belem = [i for j in belem for i in j]
            belem = (np.asarray(belem)).reshape(len(belem), 1)

            bgrp[ibgrp] = np.vstack([bgrp[ibgrp][:, 0], bgrp[ibgrp][:, 1], belem.flatten(), bface.flatten()]).T

        return bgrp

    @staticmethod
    def mesh_bgrp_sbp(nelem, bgrp, edge):
        """Includes element number and local face number to the boundary information contained in bgrp"""

        for ibgrp in range(0, len(bgrp)):
            # give all edges in the element unique identifier based on the vertices they connect
            s1 = np.char.array(edge[:, 0] * 10) + np.char.array(edge[:, 1] * 10)
            # give edges on the boundaries unique identifier based on the vertices they connect
            s2 = np.char.array(bgrp[ibgrp][:, 0] * 10) + np.char.array(bgrp[ibgrp][:, 1] * 10)
            # find indices where the boundary edge identifier matches the edeges in the element
            ind_edge = np.where(np.in1d(s1, s2))[0]

            # create a list to contain the boundary element numbers and local boundary facets numbers
            belem = list()
            bface = np.zeros((len(ind_edge), 1), dtype=int)

            # edge[0:nelem, :] --> facet 0; edge[nelem:2*nelem] --> facet 1; and edge[2*nelem:3*nelem] --> facet 2
            # (see ow the edge matrix is constructed in "mid_edge" method in mesh_generator.py). Therefore,
            # if index is below nelem, the element number doesn't change and the local face number is 0
            belem.append(ind_edge[np.where(ind_edge < nelem)])
            bface[np.where(ind_edge < nelem)] = 1 #0

            # if index is >= nelem but < 2*nelem, element number = ind_edge - nelem and local face number is 2
            belem.append(ind_edge[np.where(np.logical_and(nelem <= ind_edge, ind_edge < 2 * nelem))] - nelem)
            bface[np.where(np.logical_and(nelem <= ind_edge, ind_edge < 2 * nelem))] = 2 #1

            # if index is >= 2*nelem  but < 3*nelem, element number = ind_edge - 2*nelem and local face number is 0
            belem.append(ind_edge[np.where(np.logical_and(2 * nelem <= ind_edge, ind_edge < 3 * nelem))] - 2 * nelem)
            bface[np.where(np.logical_and(2 * nelem <= ind_edge, ind_edge < 3 * nelem))] = 0 #2

            belem = [i for j in belem for i in j]
            belem = (np.asarray(belem)).reshape(len(belem), 1)

            bgrp[ibgrp] = np.vstack([bgrp[ibgrp][:, 0], bgrp[ibgrp][:, 1], belem.flatten(), bface.flatten()]).T

        return bgrp

    @staticmethod
    def boundary_nodes(p, nelem, bgrp, vmapB, vmapM, mapB, mapM):
        nface = 3
        nfp = p+1

        # get vertex on each element
        vmapM = vmapM.reshape(nelem, nface, nfp).transpose(0, 1, 2)
        mapM = mapM.reshape((nelem, nface, nfp)).transpose(0, 1, 2)

        # get the boundary nodes that the elements in index contain at its boundary on the face contained in the bgrp
        bnodes = list()
        for i in range(0, len(bgrp)):
            vmapMgrp = vmapM[bgrp[i][:, 2], bgrp[i][:, 3], :]
            bnodes.append(vmapMgrp.flatten())

        # get boundary nodes with index of mapB
        bnodesB = list()
        for i in range(0, len(bgrp)):
            mapMgrp = mapM[bgrp[i][:, 2], bgrp[i][:, 3], :]
            bnodesB.append(mapMgrp.flatten())

        return bnodes, bnodesB

    @staticmethod
    def set_bndry(u, x, y, ax, ay, time_loc, btype, bnodes, u_bndry_fun=None):

        u_vec = (u.copy()).reshape((len(u.flatten()), 1), order='F')
        x_vec = (x.copy()).reshape((len(x.flatten()), 1), order='F')
        y_vec = (y.copy()).reshape((len(y.flatten()), 1), order='F')
        for i in range(0, len(btype)):
            bndry = btype[i]
            if bndry == 'd':
                u_vec[bnodes[i]] = u_bndry_fun(x_vec[bnodes[i]], y_vec[bnodes[i]], ax, ay, time_loc)

        u0 = u_vec.reshape(u.shape, order='F')
        return u0

    @ staticmethod
    def bgrp_by_type(btype, bgrp):
        """Gets the boundary groups by type, i.e., Dirichlet or Neumann. It returns the element and facet numbers
        associated with these boundary types"""
        bgrpDs = list()
        bgrpNs = list()
        for i in range(0, len(btype)):
            bndry = btype[i]
            if bndry == 'd' or bndry == 'D' or bndry =='Dirichlet':
                bgrpDs.append(bgrp[i].astype(int))
            elif bndry == 'n' or bndry == 'N' or bndry == 'Neumann':
                bgrpNs.append(bgrp[i].astype(int))

        if bgrpDs !=[]:
            bgrpD = np.vstack(bgrpDs)[:, 2:4]
        else:
            bgrpD = bgrpDs
        if bgrpNs !=[]:
            bgrpN = np.vstack(bgrpNs)[:, 2:4]
        else:
            bgrpN = bgrpNs

        return {'bgrpD': bgrpD, 'bgrpN': bgrpN}

    @ staticmethod
    def bndry_list(btype, bnodes, bnodesB):
        vmapDs = list()
        mapDs = list()
        vmapNs = list()
        mapNs = list()

        for i in range(0, len(btype)):
            bndry = btype[i]
            if bndry == 'd' or bndry == 'D':
                vmapDs.append(bnodes[i])
                mapDs.append(bnodesB[i])

            elif bndry == 'n' or bndry == 'N':
                vmapNs.append(bnodes[i])
                mapNs.append(bnodesB[i])

        if vmapDs != []:
            vmapD = np.hstack(vmapDs)
            mapD = np.hstack(mapDs)
        else:
            vmapD = vmapDs
            mapD = mapDs
        if vmapNs !=[]:
            vmapN = np.hstack(vmapNs)
            mapN = np.hstack(mapNs)
        else:
            vmapN = vmapNs
            mapN = mapNs

        return {'mapD': mapD, 'vmapD': vmapD, 'mapN': mapN, 'vmapN': vmapN}

    @staticmethod
    def hrefine_uniform_2d(rhs_data, bL, bR, bB, bT):
        # unpack data
        rdata = SimpleNamespace(**rhs_data)
        etov = rdata.etov
        etoe = rdata.etoe
        etof = rdata.etof
        vx = rdata.vx
        vy = rdata.vy
        nelem = rdata.nelem
        nface = 3

        # get the size of the domain in the x and y directions
        Lx = np.abs(bR - bL)
        Ly = np.abs(bT - bB)

        # number face centers uniquely
        v3 = np.amax([0 + nface * np.arange(0, nelem), etof[:, 0] + nface * etoe[:, 0]], axis=0)    # face 0
        v4 = np.amax([1 + nface * np.arange(0, nelem), etof[:, 1] + nface * etoe[:, 1]], axis=0)    # face 1
        v5 = np.amax([2 + nface * np.arange(0, nelem), etof[:, 2] + nface * etoe[:, 2]], axis=0)    # face 2

        # test if unique face number is given
        bgrp = rdata.bgrp
        bgrp_nodes = np.sum(len(bgrp[0])+len(bgrp[1])+len(bgrp[2])+len(bgrp[3]))
        unique_faces = int((3*nelem-bgrp_nodes)/2) + bgrp_nodes
        unique_nodes = len(np.unique(np.hstack([v3, v4, v5]).flatten()))
        test_unique_face = (unique_faces == unique_nodes)

        # renumber face centers starting from nv which is due to the already existing vertices 0 to nv,
        # where nv is the total number of vertices
        nv = int(np.max(etov)+1)
        ids = np.unique(np.hstack([v3, v4, v5]))
        newids = np.zeros((max(ids)+1, 1), dtype=int)
        newids[ids, 0] = np.arange(0, len(ids), dtype=int)

        v3 = nv + newids[v3]
        v4 = nv + newids[v4]
        v5 = nv + newids[v5]

        # get the vertices of the original triangles to be refined
        # see fig
        # the relation between element k, edge and face
        #           2
        #           |\
        #           | \
        #           |  \                --> etov[k] = [1, 2, 0]
        #         5 |   \ 4             --> etov[k, [1, 2]] = edge[2, 0] --> face 1
        #           |  k \              --> etov[k, [2, 0]] = edge[0, 1] --> face 2
        #           |     \  face0      --> etov[k, [0, 1]] = edge[1, 2] --> face 0
        #    face1  |______\
        #          0    3   1
        #            face2
        #

        v0 = etov[:, 0].reshape(len(etov[:, 0]), 1)
        v1 = etov[:, 1].reshape(len(etov[:, 1]), 1)
        v2 = etov[:, 2].reshape(len(etov[:, 2]), 1)

        # replace etov with the newly formed center triangle, i.e., (3 4 5)
        etov = np.zeros((4*nelem, 3), dtype=int)    # because 1 tringle is divided into 4
        etov[0:nelem, :] = np.hstack([v3, v4, v5])  # see fig in method mid_edge of class MeshGenerator2D

        # add the rest of the triangles at the bottom of the etov table
        # etov = np.vstack([etov, np.hstack([v0, v3, v5]), np.hstack([v3, v1, v4]), np.hstack([v5, v4, v2])])
        etov[nelem:4*nelem, 0] = np.vstack([v0, v1, v2]).flatten()
        etov[nelem:4*nelem, 1] = np.vstack([v3, v4, v5]).flatten()
        etov[nelem:4*nelem, 2] = np.vstack([v5, v3, v4]).flatten()

        # evaluate the coordinate values of each vertex
        x0 = vx[v0]
        x1 = vx[v1]
        x2 = vx[v2]
        y0 = vy[v0]
        y1 = vy[v1]
        y2 = vy[v2]

        vmax = np.max([v3, v4, v5])
        vx = np.vstack([vx.reshape(len(vx), 1), np.zeros((int(vmax) - len(vx) + 1, 1))])
        vy = np.vstack([vy.reshape(len(vy), 1), np.zeros((int(vmax) - len(vy) + 1, 1))])

        vx[v3, 0] = 0.5*(x0 + x1)
        vy[v3, 0] = 0.5*(y0 + y1)
        vx[v4, 0] = 0.5*(x1 + x2)
        vy[v4, 0] = 0.5*(y1 + y2)
        vx[v5, 0] = 0.5*(x2 + x0)
        vy[v5, 0] = 0.5*(y2 + y0)

        # vxy = np.hstack([vx.flatten(), vy.flatten()])
        vxy = np.array([vx.flatten(), vy.flatten()]).T

        # get edge
        vxy_mid, edge = MeshGenerator2D.mid_edge(vxy, etov)

        # get boundary group
        bgrp = MeshGenerator2D.get_bgrp(vxy_mid, edge, bL, bR, bB, bT)

        # update number of elements and number of vertices
        nelem = etov.shape[0]
        nvert = vxy.shape[0]
        vx = vx.flatten()
        vy = vy.flatten()

        return {'etov': etov, 'vx': vx, 'vy': vy, 'vxy': vxy, 'nelem': nelem, 'nvert': nvert, 'bgrp': bgrp,
                'edge': edge, 'Lx': Lx, 'Ly': Ly}

    @staticmethod
    def set_bndry_sbp_2D(xf, yf, bgrpD, bgrpN, bL, bR, bB, bT, uDL_fun=None, uNL_fun=None, uDR_fun=None, uNR_fun=None,
                         uDB_fun=None, uNB_fun=None, uDT_fun=None, uNT_fun=None):
        """Calculates boundary conditions at boundary facet nodes. This is for rectangular domain, it needs to be
        changed for other types of domains where both x and y axis might be required to impose the boundary
        conditions"""
        dim = 2
        nface = dim + 1
        nfp = int(xf.shape[0]/nface)
        nelem = xf.shape[1]
        tol = 1e-12

        # boundary facet nodes by facet number
        fid1 = np.arange(0, nfp)
        fid2 = np.arange(nfp, 2*nfp)
        fid3 = np.arange(2*nfp, 3*nfp)
        fid = [fid1, fid2, fid3]

        uD = np.zeros((nfp*nface, nelem))
        uN = np.zeros((nfp*nface, nelem))

        for i in range(0, len(bgrpD)):
            elem = bgrpD[i, 0]
            face = bgrpD[i, 1]
            # left boundary
            if np.abs(xf[fid[face][1], elem] - bL) <= tol:
                uD[fid[face], elem] = uDL_fun(xf[fid[face], elem], yf[fid[face], elem])
            # right boundary
            if np.abs(xf[fid[face][1], elem] - bR) <= tol:
                uD[fid[face], elem] = uDR_fun(xf[fid[face], elem], yf[fid[face], elem])
            # bottom boundary
            if np.abs(yf[fid[face][1], elem] - bB) <= tol:
                uD[fid[face], elem] = uDB_fun(xf[fid[face], elem], yf[fid[face], elem])
            # top boundary
            if np.abs(yf[fid[face][1], elem] - bT) <= tol:
                uD[fid[face], elem] = uDT_fun(xf[fid[face], elem], yf[fid[face], elem])

        for i in range(0, len(bgrpN)):
            elem = bgrpN[i, 0]
            face = bgrpN[i, 1]
            # left boundary
            if np.abs(xf[fid[face][1], elem] - bL) <= tol:
                uN[fid[face], elem] = uNL_fun(xf[fid[face], elem], yf[fid[face], elem])
            # right boundary
            if np.abs(xf[fid[face][1], elem] - bR) <= tol:
                uN[fid[face], elem] = uNR_fun(xf[fid[face], elem], yf[fid[face], elem])
            # bottom boundary
            if np.abs(yf[fid[face][1], elem] - bB) <= tol:
                uN[fid[face], elem] = uNB_fun(xf[fid[face], elem], yf[fid[face], elem])
            # top boundary
            if np.abs(yf[fid[face][1], elem] - bT) <= tol:
                uN[fid[face], elem] = uNT_fun(xf[fid[face], elem], yf[fid[face], elem])

        return uD, uN

    @staticmethod
    def connectivity_etoe2(etoe, etof):
        """Finds the neighbor of the neighbor for each element, the facets the element shares with the neighbor,
        and the facets the neighbor shares with the second neighbor"""

        # construct element to element connection with extended neighbors included (traversing counterclockwise)
        # we refer the elements as follows
        #     ek -- current element
        #     ev# -- first neighbor connected to # facet of ek
        #     eq# -- second neighbor connected to # facet of ev

        nface = 3
        nelem = etoe.shape[0]
        etoe2 = np.zeros((nelem, 9))
        etof_nbr = np.zeros((nelem, 9))
        etof2 = np.zeros((nelem, 9))

        face = 0
        for i in range(0, nface):
            # find ev connected to facet f of ek
            evf = etoe[:, i]
            etoe2[:, face] = evf

            # get facet numbers of evf
            evfn1 = etof[:, i]

            # get the remaining facet numbers
            evfn2_temp = evfn1 - 1
            evfn2 = np.where(evfn2_temp>-1, evfn2_temp, evfn2_temp+3)
            evfn3_temp = evfn1 + 1
            evfn3 = np.where(evfn3_temp<3, evfn3_temp, 0*evfn3_temp)

            # get the second neighbor elements
            etoe2[:, face+1] = etoe[evf, evfn2]
            etoe2[:, face+2] = etoe[evf, evfn3]

            face += 3

        # check for boundary faces and amend connectivity matrix
        for elem in range(0, nelem):
            if etoe2[elem, 0] == elem:
                etoe2[elem, 1] = elem
                etoe2[elem, 2] = elem
            if etoe2[elem, 3] == elem:
                etoe2[elem, 4] = elem
                etoe2[elem, 5] = elem
            if etoe2[elem, 6] == elem:
                etoe2[elem, 7] = elem
                etoe2[elem, 8] = elem

        etoe2 = etoe2.astype(int)

        etof_nbr[:, 0] = etof[:, 0]
        etof_nbr[:, 3] = etof[:, 1]
        etof_nbr[:, 6] = etof[:, 2]

        etof_nbr[:, 1] = np.where(etoe[etoe2[:, 0], 0] != etoe2[:, 1], etof_nbr[:, 1], 0)
        etof_nbr[:, 1] = np.where(etoe[etoe2[:, 0], 1] != etoe2[:, 1], etof_nbr[:, 1], 1)
        etof_nbr[:, 1] = np.where(etoe[etoe2[:, 0], 2] != etoe2[:, 1], etof_nbr[:, 1], 2)

        etof_nbr[:, 2] = np.where(etoe[etoe2[:, 0], 0] != etoe2[:, 2], etof_nbr[:, 2], 0)
        etof_nbr[:, 2] = np.where(etoe[etoe2[:, 0], 1] != etoe2[:, 2], etof_nbr[:, 2], 1)
        etof_nbr[:, 2] = np.where(etoe[etoe2[:, 0], 2] != etoe2[:, 2], etof_nbr[:, 2], 2)

        etof_nbr[:, 4] = np.where(etoe[etoe2[:, 3], 0] != etoe2[:, 4], etof_nbr[:, 4], 0)
        etof_nbr[:, 4] = np.where(etoe[etoe2[:, 3], 1] != etoe2[:, 4], etof_nbr[:, 4], 1)
        etof_nbr[:, 4] = np.where(etoe[etoe2[:, 3], 2] != etoe2[:, 4], etof_nbr[:, 4], 2)

        etof_nbr[:, 5] = np.where(etoe[etoe2[:, 3], 0] != etoe2[:, 5], etof_nbr[:, 5], 0)
        etof_nbr[:, 5] = np.where(etoe[etoe2[:, 3], 1] != etoe2[:, 5], etof_nbr[:, 5], 1)
        etof_nbr[:, 5] = np.where(etoe[etoe2[:, 3], 2] != etoe2[:, 5], etof_nbr[:, 5], 2)

        etof_nbr[:, 7] = np.where(etoe[etoe2[:, 6], 0] != etoe2[:, 7], etof_nbr[:, 7], 0)
        etof_nbr[:, 7] = np.where(etoe[etoe2[:, 6], 1] != etoe2[:, 7], etof_nbr[:, 7], 1)
        etof_nbr[:, 7] = np.where(etoe[etoe2[:, 6], 2] != etoe2[:, 7], etof_nbr[:, 7], 2)

        etof_nbr[:, 8] = np.where(etoe[etoe2[:, 6], 0] != etoe2[:, 8], etof_nbr[:, 8], 0)
        etof_nbr[:, 8] = np.where(etoe[etoe2[:, 6], 1] != etoe2[:, 8], etof_nbr[:, 8], 1)
        etof_nbr[:, 8] = np.where(etoe[etoe2[:, 6], 2] != etoe2[:, 8], etof_nbr[:, 8], 2)

        etof_nbr = etof_nbr.astype(int)

        # get the facet number with which the neighbor of the neighbor is connected with the neighbor
        # etoe2[:, i] -- gives the element numbers that the neighbor element is connected to
        # etof_nbr[:, i] -- gives the facet number with which the neighbor is connected to its neighbors
        # etof[e, f] -- gives the facet number f used by a neighbor to connected to element e
        for i in range(0, nface*nface):
            etof2[:, i] = etof[etoe2[:, i], etof_nbr[:, i]]

        etof2 = etof2.astype(int)

        return {'etoe2': etoe2, 'etof2': etof2, 'etof_nbr': etof_nbr}

    @staticmethod
    def curve_mesh2d(r, s, x, y, vx, vy, etov, p_map=2, Lx=1, Ly=1, curve_mesh=True, func=None):

        # obtain degree p Lagrange finite element nodes on reference element
        x_ref, y_ref = Ref2D_DG.nodes_2d(p_map)         # on equilateral triangle element
        r_lag, s_lag = Ref2D_DG.xytors(x_ref, y_ref)    # on right triangle reference element

        # apply affine mapping and obtain Lagrange finite element node location on the physical elements
        x_lag, y_lag = MeshTools2D.affine_map_2d(vx, vy, r_lag, s_lag, etov)

        # apply mapping to Lagrangian nodes on the physical space
        if func is not None:
            x_lag2, y_lag2 = func(x_lag, y_lag)
        else:
            # use function from Jesse Chan et.al. 2019 paper: Efficient Entropy Stable Gauss Collocation Methods
            alpha = 1/16
            x_lag2 = x_lag + Lx*alpha*np.cos(np.pi/Lx * (x_lag - Lx/2)) * np.cos(3*np.pi/Ly * y_lag)
            y_lag2 = y_lag + Ly*alpha*np.sin(4*np.pi/Lx * (x_lag2 - Lx/2)) * np.cos(np.pi/Ly * y_lag)

        # get degree p Vandermonde matrix evaluated at the Lagrange finite element nodes and the SBP nodes
        nelem = x.shape[1]
        # create matrices to store SBP nodes on curved elements
        xcurved = np.zeros(x.shape)
        ycurved = np.zeros(y.shape)
        xr = np.zeros(y.shape)
        xs = np.zeros(y.shape)
        yr = np.zeros(y.shape)
        ys = np.zeros(y.shape)

        V_lag = Ref2D_DG.vandermonde_2d(p_map, r_lag, s_lag)
        V_sbp = Ref2D_DG.vandermonde_2d(p_map, r, s)
        Vder_sbp = Ref2D_DG.grad_vandermonde2d(p_map, r, s)
        Vr_sbp = Vder_sbp['vdr']
        Vs_sbp = Vder_sbp['vds']

        if x.shape[1] != 1:
            for j in range(nelem):
                # get coefficients that give the purturbed Lagrange finite element nodes
                xhat = (np.linalg.inv(V_lag) @ x_lag2[:, j])
                yhat = (np.linalg.inv(V_lag) @ y_lag2[:, j])

                # Interpolating the Lagrange polynomial on the curved elements to get the mapped SBP nodes
                xcurved[:, j] = V_sbp @ xhat
                ycurved[:, j] = V_sbp @ yhat

                # Find geometric terms by interpolating the exact geometric terms
                xr[:, j] = Vr_sbp @ xhat
                xs[:, j] = Vs_sbp @ xhat
                yr[:, j] = Vr_sbp @ yhat
                ys[:, j] = Vs_sbp @ yhat

        else:
            x = x.reshape(x.shape[0], 1)
            y = y.reshape(y.shape[0], 1)

            if func is not None:
                xcurved, ycurved = func(x, y)
            elif curve_mesh:
                # map to curved element with out polynomial interpolation (used for plotting purposes only)
                xcurved = x + Lx * alpha * np.cos(np.pi / Lx * (x - Lx / 2)) * np.cos(3 * np.pi / Ly * y)
                ycurved = y + Ly * alpha * np.sin(4 * np.pi / Lx * (xcurved - Lx / 2)) * np.cos(np.pi / Ly * y)
            else:
                xcurved = x
                ycurved = y

        return {'x': xcurved, 'y': ycurved, 'xr': xr, 'xs': xs, 'yr': yr, 'ys': ys}

    @staticmethod
    def map_operators_to_phy_2d(p, nelem, H, Dr, Ds, Er, Es, R1, R2, R3, B1, B2, B3, rx, ry, sx, sy, jac,
                                            surf_jac, nx, ny, LB=None):
        # get number of nodes per face and number of volume nodes per elements
        nfp = p+1
        nnodes = H.shape[0]

        # face id
        fid1 = np.arange(0, nfp)
        fid2 = np.arange(nfp, 2 * nfp)
        fid3 = np.arange(2 * nfp, 3 * nfp)

        # get the geometric factors for each element (in rxB, B stands for Block), and write in block diagonal 3D matrix
        rxB = CalcTools.matrix_to_3D_block_diag(rx)
        ryB = CalcTools.matrix_to_3D_block_diag(ry)
        sxB = CalcTools.matrix_to_3D_block_diag(sx)
        syB = CalcTools.matrix_to_3D_block_diag(sy)

        # get volume and surface Jacobians for each elements
        jacB = CalcTools.matrix_to_3D_block_diag(jac)
        surf_jac1B = surf_jac[fid1, :].T.reshape(nelem, nfp, 1)
        surf_jac2B = surf_jac[fid2, :].T.reshape(nelem, nfp, 1)
        surf_jac3B = surf_jac[fid3, :].T.reshape(nelem, nfp, 1)

        surf_jacB = [surf_jac1B, surf_jac2B, surf_jac3B]

        # get the normal vectors on each facet.
        nx1B = nx[fid1, :].T.reshape((nelem, nfp, 1))
        ny1B = ny[fid1, :].T.reshape((nelem, nfp, 1))
        nx2B = nx[fid2, :].T.reshape((nelem, nfp, 1))
        ny2B = ny[fid2, :].T.reshape((nelem, nfp, 1))
        nx3B = nx[fid3, :].T.reshape((nelem, nfp, 1))
        ny3B = ny[fid3, :].T.reshape((nelem, nfp, 1))

        nxB = [nx1B, nx2B, nx3B]
        nyB = [ny1B, ny2B, ny3B]

        # get the extrapolation/interpolation matrix on each element
        R1B = np.block([R1] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
        R2B = np.block([R2] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)
        R3B = np.block([R3] * nelem).T.reshape(nelem, nnodes, nfp).transpose(0, 2, 1)

        RB = [R1B, R2B, R3B]

        # get volume norm matrix and its inverse on physical elements
        HB = jacB @ np.block([H] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        # HB = np.block([H] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)

        # get surface norm matrix for each facet of each element
        BB1 = (surf_jac1B * np.block([B1] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1))
        BB2 = (surf_jac2B * np.block([B2] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1))
        BB3 = (surf_jac3B * np.block([B3] * nelem).T.reshape(nelem, nfp, nfp).transpose(0, 2, 1))

        BB = [BB1, BB2, BB3]

        # get the derivative operator on the physical elements and store it for each element
        DrB = np.block([Dr] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        DsB = np.block([Ds] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)

        # # the derivative operator on the physical elements can be constructed in different ways, e.g., using the
        # # skew symmetric formulation Dx = H^-1 (Sx + 1/2 Ex), but here we construct it based on the accuracy analysis
        # # in Crean et.al.(2017) (Section 5.5, page 24)
        # DxB = 1/2*(rxB @ DrB + sxB @ DsB) + 1/2*np.linalg.inv(jacB) @ (DrB @ jacB @ rxB + DsB @ jacB @ sxB)
        # DyB = 1/2*(ryB @ DrB + syB @ DsB) + 1/2*np.linalg.inv(jacB) @ (DrB @ jacB @ ryB + DsB @ jacB @ syB)
        #
        # # construct E, the surface integral matrix
        # ErB = np.block([Er] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        # EsB = np.block([Es] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        # # again we use the result in the accuracy analysis of Crean et.al.(2017) (section 5.5) to construct Ex and Ey
        # # instead of the decomposition Ex = sum (Rgk.T @ B @ N @ R)
        # ExB = ErB @ (jacB @ rxB) + EsB @ (jacB @ sxB)
        # EyB = ErB @ (jacB @ ryB) + EsB @ (jacB @ syB)
        #
        # # construct Q, the weak derivative operator on the physical elements
        # QxB = HB @ DxB
        # QyB = HB @ DyB

        # construct S, the skew-symmetric matrix
        Qr = H @ Dr
        Qs = H @ Ds
        QrB = np.block([Qr] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)
        QsB = np.block([Qs] * nelem).T.reshape(nelem, nnodes, nnodes).transpose(0, 2, 1)

        SxB = 1 / 2 * ((jacB @ rxB) @ QrB + (jacB @ sxB) @ QsB) \
              - 1 / 2 * (QrB.transpose(0, 2, 1) @ (jacB @ rxB) + QsB.transpose(0, 2, 1) @ (jacB @ sxB))
        SyB = 1 / 2 * ((jacB @ ryB) @ QrB + (jacB @ syB) @ QsB) \
              - 1 / 2 * (QrB.transpose(0, 2, 1) @ (jacB @ ryB) + QsB.transpose(0, 2, 1) @ (jacB @ syB))

        # -------------------------------------------------------------------------------------------------------------
        # the alternative ways to construct the Ex and Ey matrices
        # construct E, the surface integral matrix
        ExB =   RB[0].transpose(0, 2, 1) @ (BB[0] * nxB[0]) @ RB[0] \
              + RB[1].transpose(0, 2, 1) @ (BB[1] * nxB[1]) @ RB[1] \
              + RB[2].transpose(0, 2, 1) @ (BB[2] * nxB[2]) @ RB[2]

        EyB =   RB[0].transpose(0, 2, 1) @ (BB[0] * nyB[0]) @ RB[0] \
              + RB[1].transpose(0, 2, 1) @ (BB[1] * nyB[1]) @ RB[1] \
              + RB[2].transpose(0, 2, 1) @ (BB[2] * nyB[2]) @ RB[2]


        # construct Q, the weak derivative operator on the physical elements
        QxB = SxB + 1 / 2 * ExB
        QyB = SyB + 1 / 2 * EyB

        # the alternative ways to construct the Dx and Dy
        HB_inv = np.linalg.inv(HB)
        DxB = HB_inv @ QxB
        DyB = HB_inv @ QyB

        #--------------------------------------------------------------------------------------------------------------
        # calculate Laplacian operator
        if LB is None:
            I = np.eye(nnodes)
            Lxx = sparse.block_diag([I] * nelem)
            Lyy = sparse.block_diag([I] * nelem)
            Lxy = 0*Lxx
            Lyx = 0*Lyy
            LB = np.block([[Lxx, Lxy], [Lyx, Lyy]])
        else:
            Lxx = LB[0, 0]
            Lxy = LB[0, 1]
            Lyx = LB[1, 0]
            Lyy = LB[1, 1]

        D2B = sparse.csr_matrix((np.block([sparse.block_diag(DxB), sparse.block_diag(DyB)]) @ LB
                                 @ np.block([[sparse.block_diag(DxB)], [sparse.block_diag(DyB)]]))[0, 0])
        # D2B = sparse.block_diag(DxB) @ (Lxx @ sparse.block_diag(DxB) + Lxy @ sparse.block_diag(DyB)) \
        #       + sparse.block_diag(DyB) @ (Lyx @ sparse.block_diag(DxB) + Lyy @ sparse.block_diag(DyB))


        return {'rxB': rxB, 'ryB': ryB, 'sxB': sxB, 'syB': syB, 'jacB': jacB, 'surf_jacB': surf_jacB, 'nxB': nxB,
                'nyB': nyB, 'RB': RB, 'HB': HB, 'BB': BB, 'DxB': DxB, 'DyB': DyB, 'ExB': ExB, 'EyB': EyB, 'QxB': QxB,
                'QyB': QyB, 'SxB': SxB, 'SyB': SyB, 'D2B': D2B}

# mesh = MeshGenerator2D.rectangle_mesh(0.5)
#
#
# vx = mesh['vx']
# vy = mesh['vy']
# etov = mesh['etov']
# nelem = mesh['nelem']
# bgrp0= mesh['bgrp']
# edge = mesh['edge']
#
# bgrp = MeshTools2D.mesh_bgrp(nelem, bgrp0, edge)
#
# p = 3
# n = int((p+1)*(p+2)/2)
# x_ref, y_ref = Ref2D_DG.nodes_2d(p)
#
# r, s = Ref2D_DG.xytors(x_ref, y_ref)
#
# edge_nodes = Ref2D_DG.fmask_2d(r, s, x_ref, y_ref)
# fmask = edge_nodes['fmask']
#
# connect2d = MeshTools2D.connectivity_2d(etov)
# etoe = connect2d['etoe']
# etof = connect2d['etof']
#
# x, y = MeshTools2D.affine_map_2d(vx, vy, r, s, etov)
#
# maps = MeshTools2D.buildmaps_2d(p, n, x, y, etov, etoe, etof, fmask)
# mapB = maps['mapB']
# mapM = maps['mapM']
# mapP = maps['mapP']
# vmapM = maps['vmapM']
# vmapP = maps['vmapP']
# vmapB = maps['vmapB']
#
# bnodes = MeshTools2D.boundary_nodes(p, nelem, bgrp, vmapB, vmapM)
#
#

#
# v = Ref2D_DG.vandermonde_2d(p, r, s)
#
# drvtv = Ref2D_DG.derivative_2d(p, r, s, v)
# Dr = drvtv['Dr']
# Ds = drvtv['Ds']
#
# normals = MeshTools2D.normals_2d(p, x, y, Dr, Ds, fmask)
# mesh = MeshGenerator1D.line_mesh(0, 2, 9, 10, scheme='LGL')
# etov = mesh['etov']
# x = mesh['x']
# x_ref = mesh['x_ref']
# con = MeshTools1D.connectivity_1d(etov)
# etoe =con['etoe']
# etof = con['etof']
# masks = MeshTools1D.fmask_1d(x_ref, x)
# fmask = masks['fmask']
# maps = MeshTools1D.buildmaps_1d(etoe, etof, fmask)
# print(con)