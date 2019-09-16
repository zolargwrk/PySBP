import numpy as np
import orthopy
import quadpy


class Ref1D:
    """ Collects tools for 1D reference element
        Uses the normalized-legendre vandermonde matrix"""

    @staticmethod
    def vandermonde_1d(p, x_ref):
        """ Calculates the vandermonde matrix in 1D"""
        v = np.polynomial.legendre.legvander(x_ref, p)
        for i in range(0, p+1):
            v[:, i] /= np.sqrt(2/(2*i+1))

        return v

    @staticmethod
    def grad_vandermonde_1d(p, x_ref):
        """Calculates the gradient of the vandermonde matrix in 1D"""
        vx = np.zeros((len(x_ref), p+1))
        for i in range(1, p+1):
            jacobi_polynomial = orthopy.line_segment.tree_jacobi(x_ref, i-1, 1, 1, 'normal', symbolic=False)
            jacobi_polynomial = np.asarray(jacobi_polynomial).T
            vx[:, i] = np.sqrt(i*(i+1))*np.asarray(jacobi_polynomial)[:, i-1]
        return vx

    @staticmethod
    def derivative_1d(p, x_ref):
        """Returnes the derivative operator in 1D"""
        v = Ref1D.vandermonde_1d(p, x_ref)
        vx = Ref1D.grad_vandermonde_1d(p, x_ref)
        d_mat_ref = vx @ np.linalg.inv(v)
        return d_mat_ref

    @staticmethod
    def e_mat_1d(tl, tr):
        e_mat = tr @ tr.T - tl @ tl.T
        return e_mat

    @staticmethod
    def projectors_1d(xl_elem, xr_elem, x_ref, **kwargs):
        """Construct the boundary projection matrices
        Inputs: p   - degree of operator
                xl_elem  - left end point of the element
                xr_elem  - right end point of the element
                x_ref    - 1D mesh on reference element
                kwargs: scheme = 'LG'  - Legendre-Gauss
                        scheme = 'LGR' - Legendre-Gauss-Radau
                        leave blank for other schemes
        Output: tl - left projection matrix
                tr - right projection matrix"""

        m = len(x_ref)
        tl = np.zeros((m, 1))
        tr = np.zeros((m, 1))

        if ('LG' in list(kwargs.values())) or ('LGR' in list(kwargs.values())):
            for i in range(0, m):
                tl[i] = Ref1D.lagrange(i, xl_elem, x_ref)
                tr[i] = Ref1D.lagrange(i, xr_elem, x_ref)
        else:
            tl[0] = 1
            tr[m-1] = 1

        return {'tl': tl, 'tr': tr}

    @staticmethod
    def lagrange(p, x0, x_ref):
        """Evaluates the i-th Lagrange polynomial at x0 based on grid data x_ref
        Inputs: x0 - point at which we want to evaluate the Lagrange polynomial
                p  - degree of the Lagrange polynomial
                x_ref  - 1D mesh
        Output: y  - Lagrange polynomial value at point x0
        """
        m = len(x_ref)
        y = 1.
        for j in range(0, m):
            if p != j:
                y *= (x0 - x_ref[j]) / (x_ref[p] - x_ref[j])
        return y

    @staticmethod
    def lift_1d(tl, tr, quad_type, v=1, h_mat_ref=1):
        if quad_type == 'LG' or quad_type == 'LGL-Dense':
            e_mat_dg = np.column_stack((tl, tr))
            lift = v @ (v.T @ e_mat_dg)
        else:
            e_mat = np.column_stack((tl, tr))
            lift = np.linalg.inv(h_mat_ref) @ e_mat
        return lift

    @staticmethod
    def fmask_1d(x_ref, x, tl, tr):
        n = len(x_ref)
        nelem = int(len(x) / n)
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


class Ref2D:

    @staticmethod
    def warp_factor(p, rout):
        """Takes in equidistant nodes constructed on equilateral triangle and returns warped nodes based
        on LGL nodes.
        Inputs: p    - degree of operator
                rout - barycentric expression for nodes on the edge of interest
                       e.g., for edge 1 (see page 176 Fig. 6.4 of Nodal DG book by Hesthaven) we can
                            express any node as: rout = lambda_3 - lambda_2
                output: warp - warped nodes"""

        rout = rout.reshape(len(rout))
        # compute equidistant and LGL nodes along the edge of interest
        n = p+1
        req = np.linspace(-1, 1, n)
        rLGL = (quadpy.line_segment.gauss_lobatto(n)).points

        # compute the Vandermonde matrix based on req
        v = Ref1D.vandermonde_1d(p, req)

        # compute the normalized Legendre polynomials for all nodes up to degree p, i.e. we get p_mat = n x nr where nr
        # is the length of rout
        nr = len(rout)
        p_mat = np.zeros((n, nr))

        # for i in range(0, n):
        p_mat = np.asarray((orthopy.line_segment.tree_jacobi(rout, n-1, 0, 0, 'normal', symbolic=False)))

        # compute the Lagrange matrix
        lagr_mat = np.linalg.inv(v.T) @ p_mat

        # warp
        warp = lagr_mat.T @ (rLGL - req)

        # make warp zero at vertices (i.e., r +or- 1) and divide by the factor 1-rout^2
        warp = np.divide(warp, (1-rout**2), out=np.zeros_like(warp), where=(1-rout**2) != 0)
        warp = warp.reshape(len(warp), 1)

        return warp

    @ staticmethod
    def nodes_2d(p):
        """Computes the nodal location on equilateral triangle for a degree p operator.
        Input: p - degree of operator
        Outputs: x, y - coordinates of the nodes"""

        # total number of degrees of freedom
        nd = int((p+1)*(p+2)/2)

        # equidistant nodes on the equilateral triangle using barycentric coordinates
        lambda_1 = np.zeros((nd, 1))
        lambda_3 = np.zeros((nd, 1))

        k = 0
        for i in range(0, p+1):
            for j in range(0, p+2-(i+1)):
                lambda_1[k, 0] = i/p
                lambda_3[k, 0] = j/p
                k += 1

        lambda_2 = 1 - lambda_1 - lambda_3

        # x and y coordinates for equidistant nodes, note: x = lambda_1*v1(x) +  lambda_2*v2(x) +  lambda_3*v3(x)
        x = lambda_2*(-1) + lambda_3*1 + lambda_1*0
        y = lambda_2*(-1/np.sqrt(3)) + lambda_3*(-1/np.sqrt(3)) + lambda_1*(2/np.sqrt(3))

        # blending function at nodes on the edges
        b1 = 4*lambda_2*lambda_3
        b2 = 4*lambda_1*lambda_3
        b3 = 4*lambda_1*lambda_2

        # evaluate the warp
        warp1 = Ref2D.warp_factor(p, lambda_3 - lambda_2)
        warp2 = Ref2D.warp_factor(p, lambda_1 - lambda_3)
        warp3 = Ref2D.warp_factor(p, lambda_2 - lambda_1)

        # warp and blend
        w1 = warp1 * b1
        w2 = warp2 * b2
        w3 = warp3 * b3

        # x and y coordinates of the warped and blended nodes
        x = x + w1 + np.cos(2*np.pi/3)*w2 + np.cos(4*np.pi/3)*w3
        y = y + 0*w1 + np.sin(2*np.pi/3)*w2 + np.sin(4*np.pi/3)*w3

        x_ref = x
        y_ref = y

        return x_ref, y_ref

    @staticmethod
    def xytors(x, y):
        """Maps the coordinates of the nodes on equilateral reference triangle to a right reference triangle
        Inputs:  x, y   - coordinates on the reference equilateral triangle
        Outputs: r, s   - coordinates on the reference equilateral triangle"""

        # using the relation (x,y) = -(r+s)/2 * v1 + (r+1)/2 * v2 + (s+1)/2 * v3 where the vertices v1 = (-1, -1/sqrt(3)
        # v2 = (1, -1/sqrt(3)) and v3 = (0, 2/sqrt(3)) we get the expression for s and r in terms of x and y

        s = 1/3*((2*np.sqrt(3))*y - 1)
        r = 1/2*(2*x - 1 - s)

        return r, s

    @staticmethod
    def rstoab(r, s):
        a = np.zeros((len(r), 1))
        b = np.zeros((len(s), 1))

        for m in range(0, len(r)):
            if s[m] != 1:
                a[m] = (2 * (1 + r[m]) / (1 - s[m]) - 1)
            else:
                a[-1] = -1
        b[:, 0] = s.reshape(len(s))
        return {'a': a, 'b': b}

    @staticmethod
    def ortho_poly_simplex2d(a, b, i, j):

        h1 = orthopy.line_segment.tree_jacobi(a, i, 0, 0, 'normal', symbolic=False)[i]
        h2 = orthopy.line_segment.tree_jacobi(b, j, 2*i+1, 0, 'normal', symbolic=False)[j]

        poly = np.sqrt(2)*h1*h2*(1-b)**i
        return poly

    @staticmethod
    def vandermonde_2d(p, r, s):
        nd = int((p+1)*(p+2)/2)
        ab = Ref2D.rstoab(r, s)
        a = ab['a']
        b = ab['b']

        v = np.zeros((len(r), nd))
        k = 0
        for i in range(0, p+1):
            for j in range(0, p+1-i):
                v[:, k] = Ref2D.ortho_poly_simplex2d(a, b, i, j)[:, 0]
                k += 1

        return v

    @staticmethod
    def grad_ortho_poly_simplex2d(a, b, i, j):
        pa = orthopy.line_segment.tree_jacobi(a, i, 0, 0, 'normal', symbolic=False)[i].reshape(len(a))
        dpa = np.sqrt(i*(i+1))*(orthopy.line_segment.tree_jacobi(a, i-1, 1, 1, 'normal', symbolic=False)[i-1].reshape(len(a)))
        pb = orthopy.line_segment.tree_jacobi(b, j, 2*i+1, 0, 'normal', symbolic=False)[j].reshape(len(a))
        dpb = np.sqrt(j*(j+(2*i+1)+1))*(orthopy.line_segment.tree_jacobi(b, j-1, 2*i+1+1, 1, 'normal', symbolic=False)[j-1].reshape(len(a)))
        # np.sqrt(j*(j+(2*i+1)+1)) because np.sqrt(j*(j+(alpha-1)+(beta-1)+1))

        pa = pa.reshape(len(pa), 1)
        dpa = dpa.reshape(len(pa), 1)
        pb = pb.reshape(len(pa), 1)
        dpb = dpb.reshape(len(pa), 1)

        if i > 0:
            dpsi_dr = 2*np.sqrt(2)*(1-b)**(i-1)*dpa*pb
            dpsi_ds = np.sqrt(2)*(1 + a)*(1-b)**(i-1)*dpa*pb - i*np.sqrt(2)*(1-b)**(i-1)*pa*pb + np.sqrt(2)*pa*dpb*(1-b)**i
        else:
            dpsi_dr = 2*np.sqrt(2)*dpa*pb
            dpsi_ds = np.sqrt(2)*(1 + a)*dpa*pb + np.sqrt(2)*pa*dpb

        return {'dpsi_dr': dpsi_dr, 'dpsi_ds': dpsi_ds}

    @staticmethod
    def grad_vandermonde2d(p, r, s):
        nd = int((p+1)*(p+2)/2)
        vdr = np.zeros((len(r), nd))
        vds = np.zeros((len(r), nd))

        ab = Ref2D.rstoab(r, s)
        a = ab['a']
        b = ab['b']

        k = 0
        for i in range(0, p+1):
            for j in range(0, p+1-i):
                grad_ortho = Ref2D.grad_ortho_poly_simplex2d(a, b, i, j)
                vdr[:, k] = grad_ortho['dpsi_dr'].reshape(len(a))
                vds[:, k] = grad_ortho['dpsi_ds'].reshape(len(a))
                k += 1

        return {'vdr': vdr, 'vds': vds}

    @staticmethod
    def derivative_2d(p, r, s, v):
        vd = Ref2D.grad_vandermonde2d(p, r, s)
        vdr = vd['vdr']
        vds = vd['vds']
        Dr = vdr @ np.linalg.inv(v)
        Ds = vds @ np.linalg.inv(v)

        return Dr, Ds

    @staticmethod
    def fmask_2d(r, s, x, y):
        fmask0 = ((np.abs(s + 1) < 1e-8).nonzero())[0]
        fmask1 = ((np.abs(r + s) < 1e-8).nonzero())[0]
        fmask2 = ((np.abs(r + 1) < 1e-8).nonzero())[0]
        fmask = np.array([fmask0, fmask1, fmask2]).T
        fmask_list = (np.hstack([fmask1, fmask2, fmask0]))
        nface = 3

        # coordinate of nodes on the edges 1, 2, and 0 (ordered as 1, 2, 0 along each column)
        fx = x[fmask_list, :]
        fy = y[fmask_list, :]

        # coordinates of the triangle vertices 1, 2, 0 (ordered as 1, 2, 0 along each column)
        # get the indices corresponding to the vertex of the triangles
        indx0 = np.arange(0, len(fx)+1, int(len(fx)/nface))
        indx0[1:] = indx0[1:] - 1
        indx1 = indx0[1: len(indx0)-1]+1
        indx = np.sort(np.hstack([indx0, indx1]))

        fvx = fx[indx, :]
        fvy = fy[indx, :]

        # calculate distance between vertices of the triangles
        distance_vertices12 = np.sqrt((fvx[1, :] - fvx[0, :])**2 + (fvy[1, :] - fvy[0, :])**2)
        distance_vertices20 = np.sqrt((fvx[3, :] - fvx[2, :])**2 + (fvy[3, :] - fvy[2, :])**2)
        distance_vertices01 = np.sqrt((fvx[5, :] - fvx[4, :])**2 + (fvy[5, :] - fvy[4, :])**2)
        distance_vertices = np.vstack([distance_vertices12, distance_vertices20, distance_vertices01])

        return {'fx': fx, 'fy': fy, 'fvx': fvx, 'fvy': fvy, 'distance_vertices': distance_vertices, 'fmask': fmask}

    @staticmethod
    def lift_2d(p, r, s, fmask):
        # construct empty E matrix
        nfp = p+1
        nface = 3
        n = len(r)
        e_mat = np.zeros((n, nfp*nface))

        # evaluate the vandermonde matrix at face 0
        v_1d = Ref1D.vandermonde_1d(p, r[fmask[:, 0]].flatten())
        # compute mass matrix and E matrix at face 0
        mass_f0 = np.linalg.inv(v_1d @ v_1d.T)
        e_mat[fmask[:, 0].reshape(nfp, 1), np.arange(0, nfp)] = mass_f0

        # evaluate the vandermonde matrix at face 1
        v_1d = Ref1D.vandermonde_1d(p, s[fmask[:, 1]].flatten())
        # compute mass matrix and E matrix at face 1
        mass_f1 = np.linalg.inv(v_1d @ v_1d.T)
        e_mat[fmask[:, 1].reshape(nfp, 1), np.arange(nfp, 2*nfp)] = mass_f1

        # evaluate the vandermonde matrix at face 2
        v_1d = Ref1D.vandermonde_1d(p, s[fmask[:, 2]].flatten())
        # compute mass matrix and E matrix at face 2
        mass_f2 = np.linalg.inv(v_1d @ v_1d.T)
        e_mat[fmask[:, 2].reshape(nfp, 1), np.arange(2*nfp, 3*nfp)] = mass_f2

        # compute the 2D vandermonde matrix
        v = Ref2D.vandermonde_2d(p, r, s)
        # compute lift
        lift = v @ (v.T @ e_mat)

        return lift

    @staticmethod
    def compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y=1, u_z=1):
        # compute derivative in x, y, and z directions with respect to r and s
        uxr = Dr @ u_x
        uxs = Ds @ u_x

        if np.shape(u_y) == ():
            uyr = 0
            uys = 0
        else:
            uyr = Dr @ u_y
            uys = Ds @ u_y

        if np.shape(u_z) == ():
            uzr = 0
            uzs = 0
        else:
            uzr = Dr @ u_z
            uzs = Ds @ u_z

        # evaluate geometric factors
        xr = Dr @ x
        xs = Ds @ x
        yr = Dr @ y
        ys = Ds @ y
        jac = -xs * yr + xr * ys
        rx = ys / jac
        sx = -yr / jac
        ry = -xs / jac
        sy = xr / jac

        return {'uxr': uxr, 'uxs': uxs, 'uyr': uyr, 'uys': uys, 'uzr': uzr, 'uzs': uzs, 'xr': xr, 'xs':xs, 'yr': yr,
                'ys': ys, 'jac': jac, 'rx': rx, 'sx': sx, 'ry': ry, 'sy': sy}

    @staticmethod
    def gradient_2d(x, y, Dr, Ds, u_x, u_y=1, u_z=1):

        # compute necessary derivatives and geometric factors
        der_geom = Ref2D.compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y, u_z)
        uxr = der_geom['uxr']
        uxs = der_geom['uxs']
        rx = der_geom['rx']
        sx = der_geom['sx']
        ry = der_geom['ry']
        sy = der_geom['sy']

        # evaluate gradient
        ux = rx*uxr + sx*uxs
        uy = ry*uxr + sy*uxs

        return ux, uy

    @staticmethod
    def divergence_2d(x, y, Dr, Ds, u_x, u_y=1, u_z=1):
        # compute necessary derivatives and geometric factors
        der_geom = Ref2D.compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y, u_z)
        uxr = der_geom['uxr']
        uxs = der_geom['uxs']
        uyr = der_geom['uyr']
        uys = der_geom['uys']
        rx = der_geom['rx']
        sx = der_geom['sx']
        ry = der_geom['ry']
        sy = der_geom['sy']

        # evaluate divergence
        divu = rx * uxr + sx * uxs + ry * uyr + sy * uys

        return divu

    @staticmethod
    def curl_2d(x, y, Dr, Ds, u_x, u_y=1, u_z=1):
        # compute necessary derivatives and geometric factors
        der_geom = Ref2D.compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y, u_z)
        uxr = der_geom['uxr']
        uxs = der_geom['uxs']
        uyr = der_geom['uyr']
        uys = der_geom['uys']
        uzr = der_geom['uzr']
        uzs = der_geom['uzs']
        rx = der_geom['rx']
        sx = der_geom['sx']
        ry = der_geom['ry']
        sy = der_geom['sy']

        # evaluate curl
        curlx = ry * uzr + sy * uzs
        curly = -rx * uzr - sx * uzs
        curlz = rx * uyr + sx * uys - ry * uxr - sy * uxs

        return {'curlx': curlx, 'curly': curly, 'curlz': curlz}

#
# p = 3
# n = int((p+1)*(p+2)/2)
# kk = Ref2D.nodes_2d(p)
# x_ref = kk['x_ref']
# y_ref = kk['y_ref']
# rs = Ref2D.xytors(x_ref, y_ref)
# r = rs['r']
# s = rs['s']
# edge_nodes = Ref2D.fmask_2d(r, s, x_ref, y_ref)
# fmask = edge_nodes['fmask']
#
#
# v = Ref2D.vandermonde_2d(p, r, s)
#
# drvtv = Ref2D.derivative_2d(p, r, s, v)
# Dr = drvtv['Dr']
# Ds = drvtv['Ds']
#
#
# lift = Ref2D.lift_2d(p, r, s, fmask)