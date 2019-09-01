import quadpy
import numpy as np
import warnings


class MeshGenerator:
    """Contains methods that create meshes for 1D, 2D, and 3D implementations"""

    @staticmethod
    def line_mesh(xl, xr, n, nelem, quad_type=0, x_ref=[]):
        """Creates equidistance nodes for a given interval
        Inputs: xl - left end point
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
            x_ref = np.linspace(-1, 1, n)
        elif quad_type == 'HGTL':
            x_ref = x_ref
            if x_ref == []:
                raise Exception("Please provide reference element: x_ref is missing.")
            if x_ref[0] != -1:
                warnings.warn("It looks like x_ref is not HGT type, it should include boundary nodes and should be "
                              "defined on [-1, 1].")
        elif quad_type == 'HGT':
            x_ref = x_ref
            if x_ref == []:
                raise Exception("Please provide reference element: x_ref is missing.")
            if x_ref[0] == -1:
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

        return {'x': x, 'etov': etov, 'x_ref': x_ref, 'bgrp': bgrp, 'coord_elem': coord_elem}


# mesh = MeshGenerator()
# x = mesh.line_mesh(0, 2, 9, 10, scheme='LGL')

# coord = x[0]
# convty_id = x[2]
# bgrp = x[3]
# coord_elem = x[4]
# convty = x[1]
# # print(x[1])
# # print(x[2])
# # print(x[3])
