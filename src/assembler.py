import numpy as np
import quadpy
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from mesh.mesh_generator import MeshGenerator1D, MeshGenerator2D
from src.ref_elem import Ref1D, Ref2D_DG
from src.csbp_type_operators import CSBPTypeOperators


class Assembler:
    def __init__(self, p, quad_type=None, boundary_type=None):
        self.p = p
        self.quad_type = quad_type
        self.boundary_type = boundary_type

    def assembler_1d(self, xl, xr, a, nelem, n, b=1, app=1):
        quad_type = self.quad_type
        boundary_type = self.boundary_type
        nface = 2

        # # b: is the variable coefficient for second derivative problems
        # if b == 1:
        #     b = np.eye(n)
        # app: is set to 1 for first derivative twice, 2 for order-matched and compatible operators
        v = 0       # vandermonde matrix
        h_mat = 0   # H norm (Mass) matrix
        tl = 0      # left projector
        tr = 0      # right projector
        x = 0       # physical domain mesh
        x_ref = 0   # reference element mesh
        d_mat = 0   # D1 - derivative matrix
        etov = 0    # element to vertex connectivity
        db_mat = 0  # normal derivative at the boundary
        d2_mat = 0  # order-matched and compatible 2nd derivative operator

        if quad_type == 'LG' or quad_type == 'LGL-Dense':
            # number of nodes and faces per element
            n = self.p + 1
            # obtain mesh information
            mesh = MeshGenerator1D.line_mesh(self.p, xl, xr, n, nelem, quad_type)
            x = mesh['x']
            etov = mesh['etov']
            x_ref = mesh['x_ref']

            # left and right projector operators
            projectors = Ref1D.projectors_1d(-1, 1, x_ref, scheme=self.quad_type)
            tl = projectors['tl']
            tr = projectors['tr']

            # vandermonde matrix
            v = Ref1D.vandermonde_1d(self.p, x_ref)
            # derivative operator on reference mesh
            d_mat = Ref1D.derivative_1d(self.p, x_ref)
            # mass matrix (norm matrix)
            h_mat = np.linalg.inv(v @ v.T)

            db_mat = d_mat
            d2_mat = d_mat @ d_mat

        elif quad_type == 'LGL':
            # number of nodes and faces per element
            n = self.p + 1
            # obtain mesh information
            mesh = MeshGenerator1D.line_mesh(self.p, xl, xr, n, nelem, quad_type)
            x = mesh['x']
            etov = mesh['etov']
            x_ref = mesh['x_ref']

            # left and right projector operators
            projectors = Ref1D.projectors_1d(-1, 1, x_ref, scheme=self.quad_type)
            tl = projectors['tl']
            tr = projectors['tr']

            # vandermonde matrix
            v = Ref1D.vandermonde_1d(self.p, x_ref)
            # derivative operator on reference mesh
            d_mat = Ref1D.derivative_1d(self.p, x_ref)
            scheme = quadpy.line_segment.gauss_lobatto(n)
            wq = scheme.weights
            h_mat = np.diag(wq)

            db_mat = d_mat
            d2_mat = d_mat @ d_mat

        elif quad_type == 'CSBP':
            opers = CSBPTypeOperators.hqd_csbp(self.p, -1, 1, n, b, app)
            d_mat = opers['d_mat_ref']
            h_mat = opers['h_mat_ref']
            tl = opers['tl']
            tr = opers['tr']
            x_ref = opers['x_ref']
            db_mat = opers['db_mat_ref']
            d2_mat = opers['d2p_ref']

            # obtain mesh information
            mesh = MeshGenerator1D.line_mesh(self.p, xl, xr, n, nelem, quad_type, b, app)
            x = mesh['x']
            etov = mesh['etov']

        elif quad_type == 'CSBP_Mattsson2004':
            opers = CSBPTypeOperators.hqd_csbp_mattsson2004(self.p, -1, 1, n, b, app)
            d_mat = opers['d_mat_ref']
            h_mat = opers['h_mat_ref']
            tl = opers['tl']
            tr = opers['tr']
            x_ref = opers['x_ref']
            db_mat = opers['db_mat_ref']
            d2_mat = opers['d2p_ref']

            # obtain mesh information
            mesh = MeshGenerator1D.line_mesh(self.p, xl, xr, n, nelem, quad_type, b, app)
            x = mesh['x']
            etov = mesh['etov']

        elif quad_type == 'HGTL':
            opers = CSBPTypeOperators.hqd_hgtl(self.p, -1, 1, n, b, app)
            d_mat = opers['d_mat_ref']
            h_mat = opers['h_mat_ref']
            tl = opers['tl']
            tr = opers['tr']
            x_ref = opers['x_ref']
            db_mat = opers['db_mat_ref']
            d2_mat = opers['d2p_ref']

            # obtain mesh information
            mesh = MeshGenerator1D.line_mesh(self.p, xl, xr, n, nelem, quad_type, b, app)
            x = mesh['x']
            etov = mesh['etov']

        elif quad_type == 'HGT':
            opers = CSBPTypeOperators.hqd_hgt(self.p, -1, 1, n, b, app)
            d_mat = opers['d_mat_ref']
            h_mat = opers['h_mat_ref']
            tl = opers['tl']
            tr = opers['tr']
            x_ref = opers['x_ref']
            db_mat = opers['db_mat_ref']
            d2_mat = opers['d2p_ref']

            # obtain mesh information
            mesh = MeshGenerator1D.line_mesh(self.p, xl, xr, n, nelem, quad_type, b, app)
            x = mesh['x']
            etov = mesh['etov']

        # surface integral on the reference element
        lift = Ref1D.lift_1d(tl, tr, quad_type, v, h_mat)

        # mapping jacobian from reference to physical elements
        jac_mapping = MeshTools1D.geometric_factors_1d(x, d_mat)
        rx = jac_mapping['rx']
        jac = jac_mapping['jac']

        # edge node location
        masks = Ref1D.fmask_1d(x_ref, x, tl, tr)
        fx = masks['fx']
        fmask = masks['fmask']

        # surface normals and inverse metric at surface
        nx = MeshTools1D.normals_1d(nelem)
        fscale = 1./jac[fmask, :]

        # build connectivity matrices
        connect = MeshTools1D.connectivity_1d(etov)
        etoe = connect['etoe']
        etof = connect['etof']

        # build connectivity maps
        maps = MeshTools1D.buildmaps_1d(n, x, a, etoe, etof, fmask, boundary_type)
        vmapM = maps['vmapM']
        vmapP = maps['vmapP']
        vmapB = maps['vmapB']
        mapB = maps['mapB']
        mapI = maps['mapI']
        mapO = maps['mapO']
        vmapI = maps['vmapI']
        vmapO = maps['vmapO']

        # normals at the faces
        nx = MeshTools1D.normals_1d(nelem)

        return {'d_mat': d_mat, 'lift': lift, 'rx': rx, 'fscale': fscale, 'vmapM': vmapM, 'vmapP': vmapP, 'xl': xl,
                'vmapB': vmapB, 'mapB': mapB, 'mapI': mapI, 'mapO': mapO, 'vmapI': vmapI, 'vmapO': vmapO, 'xr': xr,
                'jac': jac, 'x': x, 'tl': tl, 'tr': tr, 'n': n, 'nx': nx, 'nelem': nelem, 'x_ref': x_ref, 'fx': fx,
                'h_mat': h_mat, 'db_mat': db_mat, 'd2_mat': d2_mat, 'vander': v}

    def assembler_2d(self, mesh):
        p = self.p
        quad_type = self.quad_type
        boundary_type = self.boundary_type

        nfp = p+1
        n = int((p+1)*(p+2)/2)
        nface = 3

        # obtain mesh data on reference element
        x_ref, y_ref = Ref2D_DG.nodes_2d(p)    # on equilateral triangle element
        r, s = Ref2D_DG.xytors(x_ref, y_ref)   # on right triangle reference element

        # obtain mesh data on the physical element
        vx = mesh['vx']
        vy = mesh['vy']
        etov = mesh['etov']
        nelem = mesh['nelem']

        # apply affine mapping and obtain mesh location of all nodes on the physical element
        x, y = MeshTools2D.affine_map_2d(vx, vy, r, s, etov)

        # obtain the nodes on the edges of the triangles on the physical element
        mask = Ref2D_DG.fmask_2d(r, s, x, y)
        fx = mask['fx']
        fy = mask['fy']
        fmask = mask['fmask']

        # get derivative and mass matrices on the reference element
        v = Ref2D_DG.vandermonde_2d(p, r, s)
        Dr, Ds = Ref2D_DG.derivative_2d(p, r, s, v)
        Mmat = (np.linalg.inv(v @ v.T))

        # obtain the lift
        lift = Ref2D_DG.lift_2d(p, r, s, fmask)

        # get necessary the geometric factors
        geo = MeshTools2D.geometric_factors_2d(x, y, Dr, Ds)
        rx = geo['rx']
        ry = geo['ry']
        sx = geo['sx']
        sy = geo['sy']
        jac = geo['jac']

        # get normals and surface scaling factor
        norm = MeshTools2D.normals_2d(p, x, y, Dr, Ds, fmask)
        nx = norm['nx']
        ny = norm['ny']
        surf_jac = norm['surf_jac']
        fscale = surf_jac/jac[fmask.reshape((fmask.shape[0]*fmask.shape[1], 1), order='F'), :].reshape(surf_jac.shape)

        # build connectivity matrices
        connect = MeshTools2D.connectivity_2d(etov)
        etoe = connect['etoe']
        etof = connect['etof']

        # build connectivity maps
        maps = MeshTools2D.buildmaps_2d(p, n, x, y, etov, etoe, etof, fmask)
        mapM = maps['mapM']
        mapP = maps['mapP']
        vmapM = maps['vmapM']
        vmapP = maps['vmapP']
        vmapB = maps['vmapB']
        mapB = maps['mapB']

        # boundary groups and boundary nodes
        bgrp0 = mesh['bgrp']
        edge = mesh['edge']
        bgrp = MeshTools2D.mesh_bgrp(nelem, bgrp0, edge)
        bnodes, bnodesB = MeshTools2D.boundary_nodes(p, nelem, bgrp, vmapB, vmapM, mapB, mapM)

        return {'nfp': nfp, 'n': n, 'nface': nface, 'nelem': nelem, 'Dr': Dr, 'Ds': Ds, 'Mmat': Mmat, 'lift':lift,
                'rx': rx, 'ry': ry, 'sx': sx, 'sy': sy, 'jac': jac, 'nx': nx, 'ny': ny, 'surf_jac': surf_jac,
                'fscale': fscale, 'mapM': mapM, 'mapP': mapP, 'vmapM': vmapM, 'vmapP': vmapP, 'vmapB': vmapB,
                'mapB': mapB, 'bgrp': bgrp, 'bnodes': bnodes, 'bnodesB': bnodesB, 'x': x, 'y': y, 'fx': fx, 'fy': fy,
                'etov': etov, 'r': r, 's': s, 'etoe': etoe, 'etof': etof, 'vx': vx, 'vy': vy}


# a = Assembler(3, 'DG')
# kk = Assembler.assembler_2d(a)

# result = Assembler.assembler_1d(a, 0, 2, 10)
# d_mat = result['d_mat']