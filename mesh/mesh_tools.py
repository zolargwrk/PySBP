import numpy as np


class MeshTools:
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
        ind = MeshTools.sub2ind((nelem, nface), elem1, face1)
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
    def buildmaps_1d(x, etoe, etof, fmask, tl, tr):
        nelem = etoe.shape[0]
        nface = 2
        n = len(tl)    # number of nodes per element
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
    def jacobian_1d(x, d_mat_ref):
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

# mesh = MeshGenerator.line_mesh(0, 2, 9, 10, scheme='LGL')
# etov = mesh['etov']
# x = mesh['x']
# x_ref = mesh['x_ref']
# con = MeshTools.connectivity_1d(etov)
# etoe =con['etoe']
# etof = con['etof']
# masks = MeshTools.fmask_1d(x_ref, x)
# fmask = masks['fmask']
# maps = MeshTools.buildmaps_1d(etoe, etof, fmask)
# print(con)