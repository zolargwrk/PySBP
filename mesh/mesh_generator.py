import quadpy
import numpy as np


class MeshGenerator:
    """Contains methods that create meshes for 1D, 2D, and 3D implementations"""

    @staticmethod
    def line_mesh(xl, xr, n, nelem, **kwargs):
        """Creates equidistance nodes for a given interval
        Inputs: xl - left end point
                xr - right end point
                n  - number of nodes per element
                nelem  - number of elements
        **kwargs:   scheme = 'CC'  - Clenshaw-Curtis
                    scheme = 'LGL' - Gauss-Lagendre
                    scheme = 'LG'  - Gauss-Lobatto
                    scheme = 'LGR' - Gauss-Radau
                    scheme = 'NC'  - Newton-Cotes
                    scheme = 'Uniform' - Uniform distribution
        Output: coord       - coordinate of the nodal locations
                convty_id   - connectivity
                bgrp        - boundary vertex location ID
                coord_elem  - x coordinate of the nodes in each element
                x_ref       - coordinates of nodes on the reference element
                convty      - connectivity of elements (actual coordinate values)
        """

        # obtain the mesh distribution for the scheme of choice on reference element [-1, 1]
        if 'CC' in list(kwargs.values()):
            scheme = quadpy.line_segment.clenshaw_curtis(n)
            xp = scheme.points
        elif 'LG' in list(kwargs.values()):
            scheme = quadpy.line_segment.gauss_legendre(n)
            xp = scheme.points
        elif 'LGL' in list(kwargs.values()):
            scheme = quadpy.line_segment.gauss_lobatto(n)
            xp = scheme.points
        elif 'NC' in list(kwargs.values()):
            scheme = quadpy.line_segment.newton_cotes_closed(n)
            xp = scheme.points
        elif 'LGR' in list(kwargs.values()):
            scheme = quadpy.line_segment.gauss_radau(n)
            xp = scheme.points
        elif 'Uniform' in list(kwargs.values()):
            xp = np.linspace(-1, 1, n)
        else:
            xp = np.linspace(-1, 1, n)

        # identify the coordinate position of each element and obtain the coordinates on the mesh [xl, xr]
        ne = nelem + 1  # number of end vertices
        velem = np.linspace(xl, xr, ne)  # coordinates of the end vertices of each element
        xl_elem = velem[0:nelem]
        xr_elem = velem[1:nelem+1]
        coord_elem = np.zeros((1, nelem, n))

        # affine mapping to the physical elements
        for elem in range(0, nelem):
            coord_elem[0, elem, :] = 1/2 * (xl_elem[elem] * (1 - xp) + xr_elem[elem] * (1 + xp))

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
        x_ref = xp

        # x coordinate on the physical element
        x = coord

        # element to vertex connectivity
        etov = convty_id

        return x, etov, x_ref, bgrp, coord_elem, convty


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
