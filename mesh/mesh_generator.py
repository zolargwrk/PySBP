import quadpy
import numpy as np
import warnings
import meshzoo
import meshio
import pygmsh
import dmsh
import gmsh
from src.csbp_type_operators import CSBPTypeOperators
import scipy.io


class MeshGenerator1D:
    """Contains methods that create meshes for 1D, 2D, and 3D implementations"""

    @staticmethod
    def line_mesh(p, xl, xr, n, nelem, quad_type=0, b=1, app=1):
        """Creates equidistance nodes for a given interval
        Inputs: p  - degree of operator
                xl - left end point
                xr - right end point
                n  - number of nodes per element
                nelem  - number of elements
        quad_types: 'CC'  - Clenshaw-Curtis
                    'LGL' - Gauss-Lagendre
                    'LG'  - Gauss-Lobatto
                    'LGR' - Gauss-Radau
                    'NC'  - Newton-Cotes
                    'Uniform' - Uniform distribution
                    'CSBP' - Uniform distribution
                    'HGTL' - Hybrid-Gauss-Trapezoidal-Lobatto (expects x_ref to be provided)
                    'HGT'  - Hybrid-Gauss-Trapezoidal(expects x_ref to be provided)

        Output: coord       - coordinate of the nodal locations
                convty_id   - connectivity
                bgrp        - boundary vertex location ID
                coord_elem  - x coordinate of the nodes in each element
                x_ref       - coordinates of nodes on the reference element
                convty      - connectivity of elements (actual coordinate values)
                b           - variable coefficient for 2nd derivative implmentation
        """

        # obtain the mesh distribution for the scheme of choice on reference element [-1, 1]
        if quad_type == 'CC':
            scheme = quadpy.line_segment.clenshaw_curtis(n)
            x_ref = scheme.points
        elif quad_type == 'LG':
            scheme = quadpy.line_segment.gauss_legendre(n)
            x_ref = scheme.points
        elif quad_type == 'LGL-Dense' or quad_type == 'LGL':
            scheme = quadpy.line_segment.gauss_lobatto(n)
            x_ref = scheme.points
        elif quad_type == 'NC':
            scheme = quadpy.line_segment.newton_cotes_closed(n)
            x_ref = scheme.points
        elif quad_type == 'LGR':
            scheme = quadpy.line_segment.gauss_radau(n)
            x_ref = scheme.points
        elif quad_type == 'Uniform' or quad_type == 'CSBP':
            oper = CSBPTypeOperators.hqd_csbp(p, xl, xr, n, b, app)
            x_ref = oper['x_ref']
        elif quad_type == 'CSBP_Mattsson2004':
            oper = CSBPTypeOperators.hqd_csbp_mattsson2004(p, xl, xr, n, b, app)
            x_ref = oper['x_ref']
        elif quad_type == 'HGTL':
            oper = CSBPTypeOperators.hqd_hgtl(p, xl, xr, n, b, app)
            x_ref = oper['x_ref']
            if x_ref == []:
                raise Exception("Please provide reference element: x_ref is missing.")
            if x_ref[0] != -1:
                warnings.warn("It looks like x_ref is not HGTL type, it should include boundary nodes and should be "
                              "defined on [-1, 1].")
        elif quad_type == 'HGT':
            oper = CSBPTypeOperators.hqd_hgt(p, xl, xr, n, b, app)
            x_ref = oper['x_ref']
            if x_ref == []:
                raise Exception("Please provide reference element: x_ref is missing.")
            if x_ref[0] == -1 and p != 1:
                warnings.warn("It looks like x_ref is not HGT type, it should not include boundary nodes and should"
                              "be defined on [-1, 1].")
        else:
            x_ref = np.linspace(-1, 1, n)
            warnings.warn("x_ref is uniform")

        # identify the coordinate position of each element and obtain the coordinates on the mesh [xl, xr]
        ne = nelem + 1  # number of end vertices
        velem = np.linspace(xl, xr, ne)  # coordinates of the end vertices of each element
        xl_elem = velem[0:nelem]
        xr_elem = velem[1:nelem+1]
        coord_elem = np.zeros((1, nelem, n))

        # affine mapping to the physical elements
        for elem in range(0, nelem):
            coord_elem[0, elem, :] = 1/2 * (xl_elem[elem] * (1 - x_ref) + xr_elem[elem] * (1 + x_ref))

        coord = coord_elem.reshape((nelem*n, 1))
        coord_elem = coord_elem[0, :, :].T

        # identify vertex to element connectivity
        convty = np.zeros((nelem, 2))
        convty[:, 0] = xl_elem
        convty[:, 1] = xr_elem

        convty_id = np.array([range(0, nelem), range(1, nelem+1)], np.int).T

        # boundary group
        bgrp = np.zeros((2, 1), np.int)
        bgrp[0] = 0
        bgrp[1] = nelem*n - 1

        # x coordinates on the reference element
        x_ref = x_ref

        # x coordinate on the physical element
        x = coord

        # element to vertex connectivity
        etov = convty_id

        return {'x': x, 'etov': etov, 'x_ref': x_ref, 'bgrp': bgrp, 'coord_elem': coord_elem, 'nelem': nelem}


class MeshGenerator2D:

    @staticmethod
    def triangle_mesh(n):
        points, cells = meshzoo.triangle(n)
        cells = {'triangle': cells}
        meshio.write_points_cells('tri.vtu', points, cells)
        return

    @staticmethod
    def rectangle_mesh(h, bL=-1, bR=1, bB=-1, bT=1, refine=False):
        # ----------------------
        # geo = dmsh.Rectangle(bL, bR, bB, bT)
        # left = bL #* np.pi
        # right = bR #* np.pi
        # bottom = bB #* np.pi
        # top = bT #* np.pi
        mshgenerator = 'gmsh'
        # ------------------- dmsh ------------
        if mshgenerator=='dmsh':
            geo = dmsh.Rectangle(bL, bR, bB, bT)
            vxy, etov = dmsh.generate(geo, h)

        #-------------------- meshzoo ---------
        # vxy, etov = meshzoo.rectangle(xmin=bL, xmax=bR, ymin=bB, ymax=bT, nx=int(np.ceil(2/h**2)+1),
        #                               ny=int(np.ceil(2/h**2)+1), zigzag=False)
        # vxy = vxy[:, 0:2]
        # --------------------- gmsh -------
        elif mshgenerator == 'gmsh':
            gmsh.initialize()
            gmsh.model.geo.addPoint(bL, bB, 0, h, 1)
            gmsh.model.geo.addPoint(bR, bB, 0, h, 2)
            gmsh.model.geo.addPoint(bR, bT, 0, h, 3)
            gmsh.model.geo.addPoint(bL, bT, 0, h, 4)
            gmsh.model.geo.addLine(1, 2, 1)
            gmsh.model.geo.addLine(3, 2, 2)
            gmsh.model.geo.addLine(3, 4, 3)
            gmsh.model.geo.addLine(4, 1, 4)
            gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
            gmsh.model.geo.addPlaneSurface([1], 1)
            gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
            ps = gmsh.model.addPhysicalGroup(2, [1])
            gmsh.model.setPhysicalName(2, ps, "Dirichlet top Boundary")
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            # gmsh.fltk.run()  # see mesh

            # refine mesh
            if refine:
                gmsh.model.mesh.refine()

            # get element to vertex connectivity matrix
            connect_elems = gmsh.model.mesh.getNodesByElementType(2)
            etov = connect_elems[0].reshape((-1, 3)) - np.min(connect_elems[0].reshape((-1, 3)))
            # get coordinates of the nodes
            nodes = np.hstack([connect_elems[0].reshape((-1, 1)) - np.min(connect_elems[0].reshape((-1, 1))),
                                connect_elems[1].reshape((-1, 3))])
            nodes_sorted = nodes[nodes[:, 0].argsort()]
            vxy = np.unique(nodes_sorted, axis=0)[:, 1:3]

            gmsh.finalize()

        # #------------------- gmsh ---------
        # gmsh.initialize()
        # gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, 5)
        # # gmsh.model.occ.setMeshSize(2, h)
        # gmsh.model.occ.synchronize()
        # gmsh.model.mesh.generate(2)
        #
        # # uniform mesh refinement
        # for i in range(0, nrefine):
        #     gmsh.model.mesh.refine()
        #
        # # get element to vertex connectivity matrix
        # connect_elems = gmsh.model.mesh.getNodesByElementType(2, 5)
        # etov = connect_elems[0].reshape((-1, 3)) - np.min(connect_elems[0].reshape((-1, 3)))
        # # get coordinates of the nodes
        # nodes = np.hstack([connect_elems[0].reshape((-1, 1)) - np.min(connect_elems[0].reshape((-1, 1))),
        #                     connect_elems[1].reshape((-1, 3))])
        # nodes_sorted = nodes[nodes[:, 0].argsort()]
        # vxy = np.unique(nodes_sorted, axis=0)[:, 1:3]
        # # gmsh.fltk.run()  # see mesh
        # gmsh.finalize()
        # #----------------

        # # vxy, etov = optimesh.cvt.quasi_newton_uniform_full(vxy, etov, 1.0e-10, 100)
        # # NOTE: optimesh gives rise to error when solving with h=0.4 (this is very weired, took me a whole day to figure
        # #      out that the issue for the solution divergence was the mesh, got to consider changing the mesher!)
        #-------------------------
        if mshgenerator=='test':
            mat = scipy.io.loadmat('C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\mesh\\square_mesh_data.mat')
            etov = (np.asarray(mat['etov'])-1).astype(int)
            vxy = np.asarray(mat['vxy'])
            bgrp_mat = (np.asarray(mat['bgrp']))
            bgrp = list()
            bgrp.append(((np.asarray(bgrp_mat[0][0]) - 1)).astype(int))
            bgrp.append(((np.asarray(bgrp_mat[0][1]) - 1)).astype(int))
            bgrp.append(((np.asarray(bgrp_mat[0][2]) - 1)).astype(int))
            bgrp.append(((np.asarray(bgrp_mat[0][3]) - 1)).astype(int))
        #---------------------------------------

        vx = vxy[:, 0]
        vy = vxy[:, 1]

        # number of elments and number of vertex
        nelem = etov.shape[0]
        nvert = vxy.shape[0]

        if vxy.shape[1] == 2:
            points = np.append(vxy, np.zeros((vxy.shape[0], 1)), axis=1)
        else:
            points = etov

        #-----------------------
        # cells = {'triangle': etov}
        # meshio.write_points_cells('square_4elem.vtu', points, cells)
        #-----------------------

        vxy_mid, edge = MeshGenerator2D.mid_edge(vxy, etov)
        bgrp = MeshGenerator2D.get_bgrp(vxy_mid, edge, bL, bR, bB, bT)

        return {'etov': etov, 'vx': vx, 'vy': vy, 'vxy': vxy, 'nelem': nelem, 'nvert': nvert, 'bgrp': bgrp, 'edge': edge}

    @staticmethod
    def mid_edge(vxy, etov):
        """Obtains the global vertex number of each edge and
        calculates the coordinate point at the middle of each edge"""

        nelem = etov.shape[0]
        # create boundary edge group
        edge = np.zeros((nelem * 3, 2), dtype=int)
        # the relation between element k, edge and face
        #           2
        #           |\
        #           | \
        #           |  \            --> etov[k] = [0, 1, 2]
        #      face1|   \ face0     --> etov[k, [1, 2]] = edge[1, 2] --> face 0
        #           |  k \          --> etov[k, [2, 0]] = edge[2, 0] --> face 1
        #           |     \         --> etov[k, [0, 1]] = edge[0, 1] --> face 2
        #           |______\
        #          0 face2  1
        #

        edge[0:nelem, [0, 1]] = etov[:, [1, 2]]             # face 0
        edge[nelem:2*nelem, [0, 1]] = etov[:, [2, 0]]       # face 1
        edge[2*nelem:3*nelem, [0, 1]] = etov[:, [0, 1]]     # face 2

        # calculate mid edge coordinate all edges
        edge_vec = edge.reshape((nelem*3*2, 1))
        vxy_edge = vxy[edge_vec, :].reshape((2, 3 * nelem * 2), order='F')
        vxy_mid = np.mean(vxy_edge, axis=0).reshape((3 * nelem, 2), order='F')

        return vxy_mid, edge.astype(int)

    @staticmethod
    def get_bgrp(vxy_mid, edge, bL, bR, bB, bT):
        # left = -1 * np.pi
        # right = 1 * np.pi
        # bottom = -1 * np.pi
        # top = 1 * np.pi
        # left = -1
        # right = 1
        # bottom = -1
        # top = 1

        tol = 1e-12
        bgrp = list()
        # left boundary
        i = np.abs(vxy_mid[:, 0] - bL) < tol
        bgrp.append(edge[i, :])

        # right boundary
        i = np.abs((vxy_mid[:, 0] - bR)) < tol
        bgrp.append(edge[i, :])

        # bottom boundary
        i = np.abs(vxy_mid[:, 1] - bB) < tol
        bgrp.append(edge[i, :])

        # top boundary
        i = np.abs(vxy_mid[:, 1] - bT) < tol
        bgrp.append(edge[i, :])

        return bgrp

# mesh = meshio.read('Maxwell025.neu')
# mesh = MeshGenerator2D.rectangle_mesh(0.25)

# mesh = MeshGenerator2D.triangle_mesh(11)
# x = mesh.line_mesh(0, 2, 9, 10, scheme='LGL')

# coord = x[0]
# convty_id = x[2]
# bgrp = x[3]
# coord_elem = x[4]
# convty = x[1]
# # print(x[1])
# # print(x[2])
# # print(x[3])
