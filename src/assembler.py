from mesh.mesh_tools import MeshTools
from mesh.mesh_generator import MeshGenerator
from src.ref_elem import Ref1D


class Assembler:
    def __init__(self, p, quad_type):
        self.p = p
        self.quad_type = quad_type

    def assembler_1d(self, xl, xr, nelem):
        # number of nodes and faces per element
        n = self.p+1
        quad_type = self.quad_type
        nface = 2

        # obtain mesh information
        mesh = MeshGenerator.line_mesh(xl, xr, n, nelem, scheme=quad_type)
        x = mesh['x']
        etov = mesh['etov']
        x_ref = mesh['x_ref']

        # vandermonde matrix
        v = Ref1D.vandermonde_1d(self.p, x_ref)

        # derivative operator on reference mesh
        d_mat = Ref1D.derivative_1d(self.p, x_ref)

        # left and right projector operators
        projectors = Ref1D.projectors_1d(-1, 1, x_ref, scheme=self.quad_type)
        tl = projectors['tl']
        tr = projectors['tr']

        # surface integral on the reference element
        lift = Ref1D.lift_1d(v, tl, tr)

        # mapping jacobian from reference to physical elements
        jac_mapping = MeshTools.jacobian_1d(x, d_mat)
        rx = jac_mapping['rx']
        jac = jac_mapping['jac']

        # edge node location
        masks = MeshTools.fmask_1d(x_ref, x)
        fx = masks['fx']
        fmask = masks['fmask']

        # surface normals and inverse metric at surface
        nx = MeshTools.normals_1d(nelem)
        fscale = 1./jac[fmask, :]

        # build connectivity matrices
        connect = MeshTools.connectivity_1d(etov)
        etoe = connect['etoe']
        etof = connect['etof']

        # build connectivity maps
        maps = MeshTools.buildmaps_1d(x, etoe, etof, fmask)
        vmapM = maps['vmapM']
        vmapP = maps['vmapP']
        vmapB = maps['vmapB']
        mapB = maps['mapB']
        mapI = maps['mapI']
        mapO = maps['mapO']
        vmapI = maps['vmapI']
        vmapO = maps['vmapO']

        return {'d_mat': d_mat, 'lift': lift, 'rx': rx, 'fscale': fscale, 'vmapM': vmapM, 'vmapP': vmapP,
                'vmapB': vmapB, 'mapB': mapB, 'mapI': mapI, 'mapO': mapO, 'vmapI': vmapI, 'vmapO': vmapO,
                'jac': jac, 'x': x}


# a = Assembler(8, 'LGL')
# result = Assembler.assembler_1d(a, 0, 2, 10)
# d_mat = result['d_mat']