import numpy as np
from scipy import sparse
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
    def fmask_1d(x_ref, x, tl, tr):
        n = len(x_ref)
        nelem = int(len(x)/n)
        x_ref_end = x_ref @ tr
        x_ref_0 = x_ref @ tl
        x_ref[len(x_ref) - 1] = x_ref_end
        x_ref[0] = x_ref_0
        fmask1 = ((np.abs(x_ref + 1) < 1e-12).nonzero())[0][0]
        fmask2 = ((np.abs(x_ref - 1) < 1e-12).nonzero())[0][0]
        fmask = np.array([fmask1, fmask2])

        fx = np.zeros((2, nelem))
        x = x.reshape((n, nelem), order='F')
        fx[0, :] = (x.T @ tl)[:, 0]
        fx[1, :] = (x.T @ tr)[:, 0]

        return {'fx': fx, 'fmask': fmask}

    @staticmethod
    def buildmaps_1d(n, x, etoe, etof, fmask):
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
        mapI = 0
        mapO = nelem*nface - 1
        vmapI = 0
        vmapO = nelem*n - 1

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
    def fmask_2d(r, s, x_ref, y_ref):
        fmask1 = ((np.abs(s + 1) < 1e-12).nonzero())[0]
        fmask2 = ((np.abs(r + s) < 1e-12).nonzero())[0]
        fmask3 = ((np.abs(r + 1) < 1e-12).nonzero())[0]
        fmask = (np.array([fmask1, fmask2, fmask3])).T

        fx = x_ref[fmask[:]]
        fy = y_ref[fmask[:]]

        return {'fx': fx, 'fy': fy, 'fmask': fmask}

    @staticmethod
    def connect_2d(etov):
        #number of faces, elements, and vertices
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
        spNodetoNode = np.asarray([id.reshape(nelem*nface, 1), np.array((np.arange(0, nelem*nface)).reshape((nelem*nface, 1), order='F')),
                                np.array(etoe.reshape((nelem*nface, 1),order='F')), np.array(etof.reshape((nface*nelem, 1), order='F'))])
        spNodetoNode = (spNodetoNode.reshape((nelem*nface*4, 1))).reshape((nelem*nface, 4), order='F')

        # sort by global face number (first row)
        sorted = spNodetoNode[spNodetoNode[:, 0].argsort(), ]

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
        x = 0.5*(-(r+s)*vx[va] + (1+r)*vx[vb] + (1+s)*vx[vc])
        y = 0.5*(-(r+s)*vy[va] + (1+r)*vy[vb] + (1+s)*vy[vc])

        return {'x': x, 'y': y}

    @staticmethod
    def buildmaps_2d(p, n, x, y, etov, etof, fmask):
        # n = (p+1)*(p+2)/2 : number of degrees of freedom per element
        nelem = etov.shape[0]
        nfp = p+1   # number of nodes on each facet
        nface = 3
        nodeids = (np.arange(0, n*nelem)).reshape((n, nelem), order='F')
        vmapM = np.zeros((nelem, nfp, nface), dtype=int)
        vmapP = np.zeros((nelem, nfp, nface), dtype=int)
        mapM = (np.arange(0, nelem*nfp*nface)).reshape((nelem*nfp*nface, 1))
        mapP = mapM.reshape((nelem, nfp, nface))

        for elem in range(0, nelem):
            for face in range(0, nface):
                vmapM[elem, :, face] = nodeids[fmask[:, face], elem]

        for elem in range(0, nelem):
            for face in range(0, nface):
                elem2 = etoe[elem, face]    # neighboring element
                face2 = etof[elem, face]    # neighboring face

                # reference length
                v1 = etov[elem, face]
                v2 = etov[elem, face % nface]
                x_vec = x.reshape((nelem*n, 1), order='F')
                y_vec = y.reshape((nelem*n, 1), order='F')
                d_ref = np.sqrt((x_vec[v1] - x_vec[v2])**2 + (y_vec[v1] - y_vec[v2])**2)

                # volume node number on left and right element
                vidM = vmapM[elem, :, face]
                vidP = vmapM[elem2, :, face2]

                # obtain the coordinate values of the facet nodes on the left and right element
                x1 = x_vec[vidM] @ np.ones((1, nfp), dtype=int)
                y1 = y_vec[vidM] @ np.ones((1, nfp), dtype=int)
                x2 = x_vec[vidP] @ np.ones((1, nfp), dtype=int)
                y2 = y_vec[vidP] @ np.ones((1, nfp), dtype=int)

                # calulate the distance between each nodes on the neighboring elements (distance matrix)
                distance = (x1 - x2.T)**2 + (y1 - y2.T)**2

                # find nodes sharing a coordinate (distance = 0)
                (idM, idP) = np.where(np.sqrt(distance) < 1e-10)

                # find the vertex numbers on the right element (vmapP)
                vmapP[elem, idM, face] = vidP[idP]

                # global numbering to nodes on the facet (relative to mapM which number the nodes from 0 to nfp*nface*nelem -1)
                mapP[elem, idM, face] = idP + (face2 - 1)*nfp + (elem2-1)*nface*nfp

        # reshape the maps into vectors
        vmapM = vmapM.reshape((nelem*nfp*nface, 1), order='F')
        mapM = mapM.reshape((nelem*nfp*nface, 1), order='F')
        vmapP = vmapP.reshape((nelem*nfp*nface, 1), order='F')
        mapP = mapP.reshape((nelem*nfp*nface, 1), order='F')

        # obtain list of boundary nodes
        mapB = np.where(vmapM == vmapP)[0]
        mapB = mapB.reshape((len(mapB), 1))
        vmapB = vmapM[mapB]

        return {'mapM': mapM, 'mapP': mapP, 'vmapM': vmapM, 'vmapP': vmapP, 'vmapB': vmapB, 'mapB': mapB}

mesh = MeshGenerator2D.rectangle_mesh(0.25)
vx = mesh['vx']
vy = mesh['vy']
etov = mesh['etov']

p = 3
n = int((p+1)*(p+2)/2)
kk = Ref2D.nodes_2d(p)
x_ref = kk['x_ref']
y_ref = kk['y_ref']
rs = Ref2D.xytors(x_ref, y_ref)
r = rs['r']
s = rs['s']
edge_nodes = MeshTools2D.fmask_2d(r, s, x_ref, y_ref)
fmask = edge_nodes['fmask']

connect2d = MeshTools2D.connect_2d(etov)
etoe = connect2d['etoe']
etof = connect2d['etof']

xy = MeshTools2D.affine_map_2d(vx, vy, r, s, etov)
x = xy['x']
y = xy['y']
maps = MeshTools2D.buildmaps_2d(p, n, x, y, etov, etof, fmask)
mapB = maps['mapB']

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