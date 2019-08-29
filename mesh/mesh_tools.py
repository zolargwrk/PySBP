import numpy as np
from mesh.mesh_generator import *


class MeshTools:
    """Collects methods used for operations on the meshes"""

    @staticmethod
    def normals_1d(nelem):
        nx = np.zeros((nelem, 2))
        nx[:, 0] = -1
        nx[:, 1] = 1
        return nx

    @staticmethod
    def connectivity_1d (etov):
        etov = np.array([[0, 1, 2, 4], [1, 2, 4, 3]], np.int).T
        nelem = etov.shape[0]   # number of elments
        nv = nelem+1            # number of vertices
        nface = 2               # number of face per element
        tot_face = nface*nelem  # total number of faces
        vn = np.array([[0], [1]], np.int)    # local face to local vertex connection

        ftov = np.zeros((2*nelem, nelem+1), np.int)     # face to vertix connectivity

        k = 0
        for elem in range(0, nelem):
            for face in range(0, nface):
                ftov[k, etov[elem, vn[face, 0]]] = 1
                k = k+1

        ftof = ftov @ (ftov.T) #- np.eye(tot_face)

        return ftov


mesh = MeshGenerator.line_mesh(0, 2.5, 9, 4, scheme='LGL')
etov = mesh[1]
con = MeshTools.connectivity_1d(etov)
print(con)