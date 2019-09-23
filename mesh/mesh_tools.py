import numpy as np
from types import SimpleNamespace
from mesh.mesh_generator import MeshGenerator2D
from src.ref_elem import Ref2D


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


class MeshTools2D:

    @staticmethod
    def geometric_factors_2d(x, y, Dr, Ds):
        xr = Dr @ x
        xs = Ds @ x
        yr = Dr @ y
        ys = Ds @ y
        jac = -xs*yr + xr*ys
        rx = ys/jac
        sx = -yr/jac
        ry = -xs
        sy = xr/jac

        return {'xr': xr, 'xs': xs, 'yr': yr, 'ys': ys, 'jac': jac, 'rx': rx, 'sx': sx, 'ry': ry, 'sy': sy}


    @staticmethod
    def connectivity_2d(etov):
        # number of faces, elements, and vertices
        nface = 3
        nelem = etov.shape[0]
        nvert = np.max(np.max(etov))

        # create list of faces 1, 2 and 3
        fnodes = np.array([etov[:, [0, 1]], etov[:, [1, 2]], etov[:, [2, 0]]])
        fnodes = np.sort(fnodes.reshape(3*nelem, 2), 1)
        fnodes = np.sort(fnodes[:], 1)

        # default element to element and element to face connectivity
        etoe = np.arange(0, nelem).reshape((nelem, 1)) @ np.ones((1, nface), dtype=int)
        etof = np.ones((nelem, 1), dtype=int) @ np.arange(0, nface).reshape((1, nface))

        # give unique id number for faces using their node number
        id = fnodes[:, 0]*nvert + fnodes[:, 1] + 1
        vtov = np.asarray([id.reshape(nelem*nface, 1), np.array((np.arange(0, nelem*nface)).reshape((nelem*nface, 1), order='F')),
                                np.array(etoe.reshape((nelem*nface, 1), order='F')), np.array(etof.reshape((nface*nelem, 1), order='F'))])
        vtov = (vtov.reshape((nelem*nface*4, 1))).reshape((nelem*nface, 4), order='F')

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
    def affine_map_2d(vx, vy, r, s, etov):
        va = etov[:, 0].T
        vb = etov[:, 1].T
        vc = etov[:, 2].T
        x = 0.5*(-(r+s)*(vx[va]).flatten() + (1+r)*(vx[vb]).flatten() + (1+s)*(vx[vc]).flatten())
        y = 0.5*(-(r+s)*(vy[va]).flatten() + (1+r)*(vy[vb]).flatten() + (1+s)*(vy[vc]).flatten())

        return x, y

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
                (idP, idM) = np.where(np.sqrt(distance) < 1e-10)

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

        # The normals are computed as shown in Fig. 6.1 of the Nodal DG book by Hesthaven and following his code
        # face 0
        nx[fid1, :] = fyr[fid1, :]
        ny[fid1, :] = -fxr[fid1, :]
        # face 1
        nx[fid2, :] = fys[fid2, :] - fyr[fid2, :]
        ny[fid2, :] = -fxs[fid2, :] + fxr[fid2, :]
        # face 3
        nx[fid3, :] = -fys[fid3, :]
        ny[fid3, :] = fxs[fid3, :]

        # normalize
        surf_jac = np.sqrt(nx*nx + ny*ny)
        nx = nx/surf_jac
        ny = ny/surf_jac

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
    def boundary_nodes(p, nelem, bgrp, vmapB, vmapM, mapB, mapM):
        nface = 3
        nfp = p+1

        # get vertex on each element
        vmapM = vmapM.reshape(nelem, nface, nfp).transpose(0, 1, 2)
        mapM = mapM.reshape((nelem, nface, nfp)).transpose(0, 1, 2)

        # get the boundary nodes that the elements in indx contain at its boundary on the face contained in the bgrp
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
    def bndry_list(btype, bnodes, bnodesB):
        vmapD = list()
        mapD = list()
        for i in range(0, len(btype)):
            bndry = btype[i]
            if bndry == 'd':
                vmapD.append(bnodes[i])
                mapD.append(bnodesB[i])

        vmapB = np.hstack(vmapD)
        mapB = np.hstack(mapD)

        return mapB, vmapB

    @staticmethod
    def hrefine_uniform_2d(rhs_data):
        # unpack data
        rdata = SimpleNamespace(**rhs_data)
        etov = rdata.etov
        etoe = rdata.etoe
        etof = rdata.etof
        vx = rdata.vx
        vy = rdata.vy
        nelem = rdata.nelem
        nface = 3

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
        nv = np.max(etov)+1
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
        vx = np.vstack([vx.reshape(len(vx), 1), np.zeros((vmax - len(vx) + 1, 1))])
        vy = np.vstack([vy.reshape(len(vy), 1), np.zeros((vmax - len(vy) + 1, 1))])

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
        bgrp = MeshGenerator2D.get_bgrp(vxy_mid, edge, )

        # update number of elments and number of vertex
        nelem = etov.shape[0]
        nvert = vxy.shape[0]
        vx = vx.flatten()
        vy = vy.flatten()

        return {'etov': etov, 'vx': vx, 'vy': vy, 'vxy': vxy, 'nelem': nelem, 'nvert': nvert, 'bgrp': bgrp, 'edge': edge}

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
# x_ref, y_ref = Ref2D.nodes_2d(p)
#
# r, s = Ref2D.xytors(x_ref, y_ref)
#
# edge_nodes = Ref2D.fmask_2d(r, s, x_ref, y_ref)
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
# v = Ref2D.vandermonde_2d(p, r, s)
#
# drvtv = Ref2D.derivative_2d(p, r, s, v)
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