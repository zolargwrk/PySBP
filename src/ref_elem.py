import numpy as np
from scipy.linalg import null_space
import orthopy
import quadpy
from src.cubature_rules import CubatureRules
from types import SimpleNamespace


class Ref1D:
    """ Collects tools for 1D reference element
        Uses the normalized-legendre vandermonde matrix"""

    @staticmethod
    def vandermonde_1d(p, x_ref):
        """ Calculates the vandermonde matrix in 1D"""
        x_ref = x_ref.flatten()
        v = np.polynomial.legendre.legvander(x_ref, p)
        for i in range(0, p+1):
            v[:, i] /= np.sqrt(2/(2*i+1))

        return v

    @staticmethod
    def grad_vandermonde_1d(p, x_ref):
        """Calculates the gradient of the vandermonde matrix in 1D"""
        x_ref = x_ref.flatten()
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


class Ref2D_DG:

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
        Outputs: x, y - coordinates of the nodes on the reference element"""

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
        warp1 = Ref2D_DG.warp_factor(p, lambda_3 - lambda_2)
        warp2 = Ref2D_DG.warp_factor(p, lambda_1 - lambda_3)
        warp3 = Ref2D_DG.warp_factor(p, lambda_2 - lambda_1)

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
        Outputs: r, s   - coordinates on the reference right triangle"""

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
        n = int((p+1)*(p+2)/2)
        ab = Ref2D_DG.rstoab(r, s)
        a = ab['a']
        b = ab['b']

        V = np.zeros((len(r), n))
        k = 0
        for i in range(0, p+1):
            for j in range(0, p+1-i):
                V[:, k] = Ref2D_DG.ortho_poly_simplex2d(a, b, i, j)[:, 0]
                k += 1

        return V

    @staticmethod
    def grad_ortho_poly_simplex2d(a, b, i, j):
        # We refer to the book by Hesthaven and Warburton: Nodal DG Methods, 2007 for the construction of the Legendre
        # polynomials. pages are referred as p. , equations eq. , section sec. , appendix app.

        # get the jacobi polynomial from the python package "orthopy" with alpha=0 and beta=0, see app.A, p.445(457/511)
        pa = orthopy.line_segment.tree_jacobi(a, i, 0, 0, 'normal', symbolic=False)[i].reshape(len(a))

        # get the jacobi polynomial from the python package "orthopy" with alpha=0 and beta=2*i+1
        pb = orthopy.line_segment.tree_jacobi(b, j, 2 * i + 1, 0, 'normal', symbolic=False)[j].reshape(len(a))

        # get the derivative of the jacobi polynomial with respect to a  and setting alpha=0 and beta=0
        # the property used to get the derivative is given in eq.A.2, p.445(457/511)
        dpa = np.sqrt(i*(i+1))*(orthopy.line_segment.tree_jacobi(a, i-1, 1, 1, 'normal', symbolic=False)[i-1].reshape(len(a)))

        # get the derivative of the jacobi polynomial with respect to b  and setting alpha=0 and beta=2*i+1
        # the property used to get the derivative is given in eq.A.2, p.445(457/511)
        dpb = np.sqrt(j*(j+(2*i+1)+1))*(orthopy.line_segment.tree_jacobi(b, j-1, 2*i+1+1, 1, 'normal', symbolic=False)[j-1].reshape(len(a)))

        # reshape into a column vector
        pa = pa.reshape(len(pa), 1)
        dpa = dpa.reshape(len(pa), 1)
        pb = pb.reshape(len(pa), 1)
        dpb = dpb.reshape(len(pa), 1)

        # apply scaling and get the derivative of the Legendre polynomial
        if i > 0:
            # use eq.6.6, p.173 (185/511) to evaluate the Legendre polynomial, also a = 2(1+r)/(1-s) - 1, and b = s
            # note that dpsi/dr = dpsi/da * da/dr + dpsi/db * db/dr, and da/dr = 2/(1-b), and db/dr = 0
            dpsi_dr = 2*np.sqrt(2)*(1-b)**(i-1)*dpa*pb
            
            # dpsi/ds = dpsi/da * da/ds + dpsi/db *db/ds, and da/ds = ((1+a)/2)/((1-b)/2) = (1+a)/(1-b), and db/ds = 1
            dpsi_ds = np.sqrt(2)*(1 + a)*(1-b)**(i-1)*dpa*pb - i*np.sqrt(2)*(1-b)**(i-1)*pa*pb + np.sqrt(2)*pa*dpb*(1-b)**i
        else:
            dpsi_dr = 2*np.sqrt(2)*dpa*pb
            dpsi_ds = np.sqrt(2)*(1 + a)*dpa*pb + np.sqrt(2)*pa*dpb

        return {'dpsi_dr': dpsi_dr, 'dpsi_ds': dpsi_ds}

    @staticmethod
    def laplacian_ortho_poly_simplex2d(a, b, i, j):
        # We refer to the book by Hesthaven and Warburton: Nodal DG Methods, 2007 for the construction of the Legendre
        # polynomials. pages are referred as p. , equations eq. , section sec. , appendix app.

        # get the first derivative of jacobi polynomial with alpha=0 and beta=0
        pa = orthopy.line_segment.tree_jacobi(a, i, 0, 0, 'normal', symbolic=False)[i]
        # get the first derivative of jacobi polynomial with alpha=0 and beta=0 (with out the scaling)
        dpa = orthopy.line_segment.tree_jacobi(a, i-1, 1, 1, 'normal', symbolic=False)[-1]
        # get the second derivative of jacobi polynomial with alpha=0 and beta=0 (with out the scaling)
        ddpa = orthopy.line_segment.tree_jacobi(a, i-2, 2, 2, 'normal', symbolic=False)[-1]

        # get the jacobi polynomial with alpha=0 and beta=2*i+1
        pb = orthopy.line_segment.tree_jacobi(b, j, 2*i + 1, 0, 'normal', symbolic=False)[-1]
        # get the first derivative of jacobi polynomial with alpha=0 and beta=2*i+1 (with out the scaling)
        dpb = orthopy.line_segment.tree_jacobi(b, j-1, 2*i + 2, 1, 'normal', symbolic=False)[-1]
        # get the second derivative of jacobi polynomial with alpha=0 and beta=2*i+1 (with out the scaling)
        ddpb = orthopy.line_segment.tree_jacobi(b, j-2, 2*i + 3, 2, 'normal', symbolic=False)[-1]

        # apply scaling and get the derivative of the Legendre polynomial
        if i > 1:
            # calculate d2psi/dr2 (see notes on Constructio of SBP operators)
            ddpsi_drr = 4*(1-b)**(i-2)*np.sqrt(2*i*(i-1)*(i+1)*(i+2)) * ddpa * pb

            #  calculate d2psi/ds2
            ddpsi_dss = (a+b)**2 * (1-b)**(i-2) * np.sqrt(2*i*(i-1)*(i+1)*(i+2)) * ddpa * pb \
                        + (2*a + 2) * (1-b)**(i-1) * np.sqrt(4*i*j*(i+1)*(j+2*i+2)) * dpa * dpb \
                        + (1-b)**i * np.sqrt(2*j*(j-1)*(j+2*i+2)*(j+2*i+3)) * pa * ddpb
        elif i > 0:
            # calculate d2psi/dr2 (see notes on Constructio of SBP operators)
            ddpsi_drr = 4 * np.sqrt(2 * i * (i - 1) * (i + 1) * (i + 2)) * ddpa * pb

            #  calculate d2psi/ds2
            ddpsi_dss = (a + b) ** 2 * np.sqrt(2 * i * (i - 1) * (i + 1) * (i + 2)) * ddpa * pb \
                        + (2 * a + 2) * (1 - b) ** (i - 1) * np.sqrt(4 * i * j * (i + 1) * (j + 2 * i + 2)) * dpa*dpb \
                        + (1 - b) ** i * np.sqrt(2 * j * (j - 1) * (j + 2 * i + 2) * (j + 2 * i + 3)) * pa * ddpb
        else:
            # calculate d2psi/dr2 (see notes on Constructio of SBP operators)
            ddpsi_drr = 4 * np.sqrt(2 * i * (i - 1) * (i + 1) * (i + 2)) * ddpa * pb

            #  calculate d2psi/ds2
            ddpsi_dss = (a + b) ** 2 * np.sqrt(2 * i * (i - 1) * (i + 1) * (i + 2)) * ddpa * pb \
                        + (2 * a + 2) *  np.sqrt(4 * i * j * (i + 1) * (j + 2 * i + 2)) * dpa * dpb \
                        + (1 - b) ** i * np.sqrt(2 * j * (j - 1) * (j + 2 * i + 2) * (j + 2 * i + 3)) * pa * ddpb

        return {'ddpsi_drr': ddpsi_drr, 'ddpsi_dss': ddpsi_dss}

    @staticmethod
    def grad_vandermonde2d(p, r, s):
        nd = int((p+1)*(p+2)/2)
        vdr = np.zeros((len(r), nd))
        vds = np.zeros((len(r), nd))

        ab = Ref2D_DG.rstoab(r, s)
        a = ab['a']
        b = ab['b']

        k = 0
        for i in range(0, p+1):
            for j in range(0, p+1-i):
                grad_ortho = Ref2D_DG.grad_ortho_poly_simplex2d(a, b, i, j)
                vdr[:, k] = grad_ortho['dpsi_dr'].reshape(len(a))
                vds[:, k] = grad_ortho['dpsi_ds'].reshape(len(a))
                k += 1

        return {'vdr': vdr, 'vds': vds}

    @staticmethod
    def laplacian_vandermonde2d(p, r, s):
        nd = int((p+1)*(p+2)/2)
        vdrr = np.zeros((len(r), nd))
        vdss = np.zeros((len(r), nd))

        ab = Ref2D_DG.rstoab(r, s)
        a = ab['a']
        b = ab['b']

        k = 0
        for i in range(0, p+1):
            for j in range(0, p+1-i):
                # obtain the second derivative of the Vandermonde matrix column by column
                lap_ortho = Ref2D_DG.laplacian_ortho_poly_simplex2d(a, b, i, j)
                vdrr[:, k] = lap_ortho['ddpsi_drr'].reshape(len(a))
                vdss[:, k] = lap_ortho['ddpsi_dss'].reshape(len(a))
                k += 1

        return {'vdrr': vdrr, 'vdss': vdss}

    @staticmethod
    def derivative_2d(p, r, s, v):
        vd = Ref2D_DG.grad_vandermonde2d(p, r, s)
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
        # fmask2 = ((np.abs(s + 1) < 1e-8).nonzero())[0]
        # fmask0 = ((np.abs(r + s) < 1e-8).nonzero())[0]
        # fmask1 = ((np.abs(r + 1) < 1e-8).nonzero())[0]
        fmask = np.array([fmask0, fmask1, fmask2]).T
        fmask_list = (np.hstack([fmask0, fmask1, fmask2]))
        nface = 3

        # coordinate of nodes on the edges 2, 0, and 1 (ordered as 2, 0, 1 along each column)
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
        # mass_f0 = np.linalg.inv(v_1d @ v_1d.T)
        mass_f0 = np.eye(2)
        e_mat[fmask[:, 0].reshape(nfp, 1), np.arange(0, nfp)] = mass_f0

        # evaluate the vandermonde matrix at face 1
        v_1d = Ref1D.vandermonde_1d(p, s[fmask[:, 1]].flatten())
        # compute mass matrix and E matrix at face 1
        # mass_f1 = np.linalg.inv(v_1d @ v_1d.T)
        mass_f1 = np.array([[np.sqrt(2), 0], [0, np.sqrt(2)]])
        e_mat[fmask[:, 1].reshape(nfp, 1), np.arange(nfp, 2*nfp)] = mass_f1

        # evaluate the vandermonde matrix at face 2
        v_1d = Ref1D.vandermonde_1d(p, s[fmask[:, 2]].flatten())
        # compute mass matrix and E matrix at face 2
        # mass_f2 = np.linalg.inv(v_1d @ v_1d.T)
        mass_f2 = np.eye(2)
        e_mat[fmask[:, 2].reshape(nfp, 1), np.arange(2*nfp, 3*nfp)] = mass_f2

        # compute the 2D vandermonde matrix
        v = Ref2D_DG.vandermonde_2d(p, r, s)
        # compute lift
        # lift = (v @ v.T) @ e_mat
        H_inv = 1.5*np.eye(3)
        lift = H_inv @ e_mat

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
        der_geom = Ref2D_DG.compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y, u_z)
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
        der_geom = Ref2D_DG.compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y, u_z)
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
        der_geom = Ref2D_DG.compute_der_geom_ref(x, y, Dr, Ds, u_x, u_y, u_z)
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

    @staticmethod
    def quad_rule_tri(p, rule='Liu-Vinokur'):
        if p==1:
            if rule=='Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_01()
            elif rule=='Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_01()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p==2:
            if rule=='Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_02()
            elif rule=='Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_02()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 2:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_02()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_02()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 3:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_03()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_04()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 4:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_04()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_04()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 5:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_05()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_05()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 6:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_06()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_06()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 7:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_07()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_07()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        elif p == 8:
            if rule == 'Liu-Vinokur':
                scheme = quadpy.triangle.liu_vinokur_08()
            elif rule == 'Witherden-Vincent':
                scheme = quadpy.triangle.witherden_vincent_08()
            else:
                raise("Quadrature type not implemented, use either 'Liu-Vinokur' or 'Witherden-Vincent' rules")
        else:
            raise ValueError('Degree greater than 8 not implemented so far')

        wts = scheme.weights
        pts = scheme.points

        pts = np.array([pts[:, 0], pts[:, 1], pts[:, 2]]).T
        r = pts @ np.array([[-1], [1], [-1]])
        s = pts @ np.array([[-1], [-1], [1]])

        return {'weights': wts, 'r': r, 's': s}

    @staticmethod
    def mass_matrix(p):
        # get quadrature rule
        if p == 1:
            scheme = quadpy.triangle.witherden_vincent_02()
        elif p == 2:
            scheme = quadpy.triangle.witherden_vincent_04()
        else:
            raise ValueError("Degree not implemented.")

        # points are returned in barycentric coordinate from quadpy, hence get x and y on the right triangle ref element
        pts = scheme.points
        # shuffle columns of pts to have them in barycentric order lambda 2, 3, 1
        # pts = np.array([pts[:, 1], pts[:, 2], pts[:, 0]]).T
        pts = np.array([pts[:, 0], pts[:, 1], pts[:, 2]]).T
        r = pts @ np.array([[-1], [1], [-1]])
        s = pts @ np.array([[-1], [-1], [1]])

        v = Ref2D_DG.vandermonde_2d(p, r, s)

        M = (np.linalg.inv(v)).T @ np.linalg.inv(v)

        # mass matrix on nodes of Hesthaven
        x_ref, y_ref = Ref2D_DG.nodes_2d(p)

        r2, s2 = Ref2D_DG.xytors(x_ref, y_ref)
        v2 = Ref2D_DG.vandermonde_2d(p, r2, s2)

        M2 = (np.linalg.inv(v2)).T @ np.linalg.inv(v2)

        return M, M2


class Ref2D_SBP:

    @staticmethod
    def shape_tri(p, sbp_family="gamma"):
        """Calculates the shape function and its derivatives at the cubature nodes"""

        sbp_family = str.lower(sbp_family)

        # get the cubature points
        cub_data = CubatureRules.cub_tri_volume(p, sbp_family)
        cub = SimpleNamespace(**cub_data)

        # get the interpolation points on equilateral triangle reference element
        x, y = Ref2D_DG.nodes_2d(p)

        # map interpolation nodes to the right triangle reference element
        r, s = Ref2D_DG.xytors(x, y)

        # calculate Vandermonde matrix (with Legendre basis) on the interpolation nodes
        V = Ref2D_DG.vandermonde_2d(p, r, s)
        # calculate the coefficient matrix
        C = np.linalg.inv(V)

        # calculate the Vandermonde matrix on the quadrature/cubature nodes
        Vq = Ref2D_DG.vandermonde_2d(p, cub.r, cub.s)

        # evaluate the shape function at the cubature nodes
        shp = Vq @ C

        # calculate the first derivatives of the Vandermonde matrix on the cubature nodes
        Vqx_data = Ref2D_DG.grad_vandermonde2d(p, cub.r, cub.s)
        vdr = Vqx_data['vdr']
        vds = Vqx_data['vds']

        # evaluate the first derivative of the shape function at the cubature nodes
        shpx = vdr @ C
        shpy = vds @ C

        # evaluate the second derivative of the shape function at the cubature nodes
        Vqxx_data = Ref2D_DG.laplacian_vandermonde2d(p, cub.r, cub.s)
        vdrr = Vqxx_data['vdrr']
        vdss = Vqxx_data['vdss']

        # caluculte the second derivative of the shape function at the cubature nodes
        shpxx = vdrr @ C
        shpyy = vdss @ C

        return {'shp': shp, 'shpx': shpx, 'shpy': shpy, 'shpxx': shpxx, 'shpyy': shpyy}

    @staticmethod
    def nodal_sym_map2D(r, s):
        """Finds which nodes are in the x=y symmetry group for given cubature nodes. First all nodes are labeled
        and then the nodes that are symmetric with the labled nodes are found by comparing their x and y components
        with those of the labeled nodes
        Args:
            r (float64) : the x component of the cubature node locations on the right triangle reference element
            s (float64) : the y component of the cubature node locations on the right triangle reference element
        Returns:
            rs_sym_map (array): an array with mapping between nodes that are symmetric about x=y line
        """
        tol = 1e-10     # tolerance between nodes that are symmetric along the x=y line
        n = len(r)      # number of cubature nodes
        # lable nodes from 1 to n and get which ones are symmetric with them (in the second row or rs_sym_map)
        rs_sym_map = np.zeros((2, n), dtype=int)
        rs_sym_map[0, :] = range(1, n+1)

        for i in range(0, n):
            for j in range(0, n):
                if rs_sym_map[1, j] != 0:   # has been mapped earlier
                    continue
                else:
                    if np.abs(r[i]-s[j]) <= tol and np.abs(r[j]-s[i]) <= tol:
                        rs_sym_map[1, j] = rs_sym_map[0, i]

        return rs_sym_map

    @ staticmethod
    def simplex_vertices(dim):
        vert = np.zeros((dim+1, dim))
        if dim == 1:
            vert = np.array([[-1], [1]])
        elif dim == 2:
            vert = np.array([[-1, -1], [1, -1], [-1, 1]])
            # vert = np.array([[1, -1], [-1, 1], [-1, -1]])
            # vert = np.array([[0, 0], [1, 0], [0, 1]]) # this doesn't work probably due to the orthogonal
                                                        # polynomial used to construct the Vandermonde matrix

        return vert

    @ staticmethod
    def cartesian_to_barycentric2D(r, s=None, vert=None):
        """Converts Cartesian coordinate to Barycentric on the right triangle reference element."""
        dim = 1
        if s is not None:
            dim = 2
        if vert is None:
            vert = Ref2D_SBP.simplex_vertices(dim)

        rs = np.ones((len(r), dim+1))
        v = np.ones((dim+1, dim+1))
        rs[:, 0] = r.flatten()
        if dim == 2:
            rs[:, 1] = s.flatten()

        v[:, 0:dim] = vert
        b = rs @ np.linalg.inv(v)

        return b

    @staticmethod
    def barycentric_to_cartesian(b, vert):
        cart = b @ vert
        return cart

    @staticmethod
    def sym_group_map2D(r, s=None):
        """Maps the cubature node by type of symmetry group"""
        tol = 1e-10     # tolerance
        n = len(r)      # number of cubature nodes
        dim = 1
        if s is not None:
            dim = 2

        # convert cartesian coordinate to barycentric and sort by row
        bry = Ref2D_SBP.cartesian_to_barycentric2D(r, s)
        bry_sort = np.sort(bry)

        # find nodes in certain symmetry group by looking at their Barycentric coordinates, i.e., a Barycentric
        # coordinate (including permutation) (a,a,a)-->S3, (a,a,1-2a)-->S21, (a,b,1-a-b)-->S111
        sym_grp_temp = np.zeros((dim+1, n, n), dtype=int)
        sym_grp = []
        sym_grp_by_type = np.zeros((dim+1, n), dtype=int)
        S2_cnt = 0
        S11_cnt = 0
        S3_cnt = 0
        S21_cnt = 0
        S111_cnt = 0

        if dim == 1:
            for i in range(0, n):
                if np.abs(bry_sort[i, 1] - bry_sort[i, 0]) <= tol:
                    # find nodes in S2 symmetry group
                    sym_grp_by_type[0, S2_cnt] = i + 1
                    S2_cnt += 1
                else:
                    # find nodes in S11 symmetry group
                    sym_grp_by_type[1, S11_cnt] = i + 1
                    S11_cnt += 1

            max_row = np.max([S2_cnt, S11_cnt])     # get the maximum number of nodes grouped in a symmetry group
            sym_grp_by_type = sym_grp_by_type[:, 0:max_row]     # eliminate unnecessary columns

            # find nodes that are in the same type of symmetry group and are permutation of one another
            for k1 in range(0, S2_cnt):
                cnt = 0
                for k2 in range(k1, S2_cnt):
                    if np.sum(np.abs(
                            bry_sort[sym_grp_by_type[0, k1] - 1, :] - bry_sort[sym_grp_by_type[0, k2] - 1, :])) <= tol:
                        if sym_grp_by_type[0, k2] not in (sym_grp_temp[0, :, :]).flatten():
                            sym_grp_temp[0, k1, k2] = sym_grp_by_type[0, k2]
                            cnt += 1
            for k1 in range(0, S11_cnt):
                cnt = 0
                for k2 in range(k1, S11_cnt):
                    if np.sum(np.abs(
                            bry_sort[sym_grp_by_type[1, k1] - 1, :] - bry_sort[sym_grp_by_type[1, k2] - 1, :])) <= tol:
                        if sym_grp_by_type[1, k2] not in (sym_grp_temp[1, :, :]).flatten():
                            sym_grp_temp[1, k1, cnt] = sym_grp_by_type[1, k2]
                            cnt += 1

            # delete zero rows and columns
            sym_grpS2 = np.delete(sym_grp_temp[0, :, :], np.where(~(sym_grp_temp[0, :, :]).any(axis=1))[0], axis=0)
            sym_grpS2 = np.delete(sym_grpS2[:, :], np.where(~(sym_grpS2[:, :]).any(axis=0))[0], axis=1)

            sym_grpS11 = np.delete(sym_grp_temp[1, :, :], np.where(~(sym_grp_temp[1, :, :]).any(axis=1))[0], axis=0)
            sym_grpS11 = np.delete(sym_grpS11[:, :], np.where(~(sym_grpS11[:, :]).any(axis=0))[0], axis=1)

            sym_grp.append(sym_grpS2)
            sym_grp.append(sym_grpS11)

        elif dim == 2:
            for i in range(0, n):
                if np.abs(bry_sort[i, 1] - bry_sort[i, 0]) <=tol and np.abs(bry_sort[i, 2] - bry_sort[i, 0]) <= tol:
                    # find nodes in S3 symmetry group
                    sym_grp_by_type[0, S3_cnt] = i+1
                    S3_cnt += 1
                elif np.abs(bry_sort[i, 1] - bry_sort[i, 0]) <=tol or np.abs(bry_sort[i, 2] - bry_sort[i, 0]) <= tol or \
                        np.abs(bry_sort[i, 2] - bry_sort[i, 1]) <= tol:
                    # find nodes in S21 symmetry group
                    sym_grp_by_type[1, S21_cnt] = i+1
                    S21_cnt += 1
                else:
                    # find nodes in S111 symmetry group
                    sym_grp_by_type[2, S111_cnt] = i+1
                    S111_cnt += 1

            max_row = np.max([S3_cnt, S21_cnt, S111_cnt])      # get the maximum number of nodes grouped in a symmetry group
            sym_grp_by_type = sym_grp_by_type[:, 0:max_row]    # eliminate unnecessary columns

            # find nodes that are in the same type of symmetry group and are permutation of one another
            for k1 in range(0, S3_cnt):
                cnt = 0
                for k2 in range(k1, S3_cnt):
                    if np.sum(np.abs(bry_sort[sym_grp_by_type[0, k1]-1, :] - bry_sort[sym_grp_by_type[0, k2]-1, :])) <= tol:
                        if sym_grp_by_type[0, k2] not in (sym_grp_temp[0, :, :]).flatten():
                            sym_grp_temp[0, k1, k2] = sym_grp_by_type[0, k2]
                            cnt += 1
            for k1 in range(0, S21_cnt):
                cnt = 0
                for k2 in range(k1, S21_cnt):
                    if np.sum(np.abs(bry_sort[sym_grp_by_type[1, k1]-1, :] - bry_sort[sym_grp_by_type[1, k2]-1, :])) <= tol:
                        if sym_grp_by_type[1, k2] not in (sym_grp_temp[1, :, :]).flatten():
                            sym_grp_temp[1, k1, cnt] = sym_grp_by_type[1, k2]
                            cnt += 1
            for k1 in range(0, S111_cnt):
                cnt = 0
                for k2 in range(k1, S111_cnt):
                    if np.sum(np.abs(bry_sort[sym_grp_by_type[2, k1]-1, :] - bry_sort[sym_grp_by_type[2, k2]-1, :])) <= tol:
                        if sym_grp_by_type[2, k2] not in (sym_grp_temp[2, :, :]).flatten():
                            sym_grp_temp[2, k1, k2] = sym_grp_by_type[2, k2]
                            cnt += 1

            # delete zero rows and columns
            sym_grpS3 = np.delete(sym_grp_temp[0, :, :], np.where(~(sym_grp_temp[0, :, :]).any(axis=1))[0], axis=0)
            sym_grpS3 = np.delete(sym_grpS3[:, :], np.where(~(sym_grpS3[:, :]).any(axis=0))[0], axis=1)

            sym_grpS21 = np.delete(sym_grp_temp[1, :, :], np.where(~(sym_grp_temp[1, :, :]).any(axis=1))[0], axis=0)
            sym_grpS21 = np.delete(sym_grpS21[:, :], np.where(~(sym_grpS21[:, :]).any(axis=0))[0], axis=1)

            sym_grpS111 = np.delete(sym_grp_temp[2, :, :], np.where(~(sym_grp_temp[2, :, :]).any(axis=1))[0], axis=0)
            sym_grpS111 = np.delete(sym_grpS111[:, :], np.where(~(sym_grpS111[:, :]).any(axis=0))[0], axis=1)

            sym_grp.append(sym_grpS3)
            sym_grp.append(sym_grpS21)
            sym_grp.append(sym_grpS111)

        return {'sym_grp': sym_grp, 'sym_grp_by_type': sym_grp_by_type}

    @staticmethod
    def make_rperm2D(sym_grp, r, s):
        nvert = 3
        n = len(r)
        Rperm = np.zeros((nvert, n), dtype=int)
        Rperm[0, :] = range(1, n+1)

        # permutation for S3 symmetry group
        for i in range(0, sym_grp[0].shape[0]):
            Rperm[:, sym_grp[0][i, 0]] = sym_grp[0][i, 0]

        # permutation for S21 symmetry group
        for i in range(0, sym_grp[1].shape[0]):
            Rperm[0, sym_grp[1][i, :]-1] = sym_grp[1][i, [0, 1, 2]]
            Rperm[1, sym_grp[1][i, :]-1] = sym_grp[1][i, [2, 0, 1]]
            Rperm[2, sym_grp[1][i, :]-1] = sym_grp[1][i, [1, 2, 0]]

        # permutation for S111 symmetry group
        for i in range(0, sym_grp[2].shape[0]):
            Rperm[0, sym_grp[2][i, :]-1] = sym_grp[2][i, [0, 1, 2, 3, 4, 5]]
            Rperm[1, sym_grp[2][i, :]-1] = sym_grp[2][i, [2, 0, 1, 5, 3, 4]]
            Rperm[2, sym_grp[2][i, :]-1] = sym_grp[2][i, [1, 2, 0, 4, 5, 3]]

        return Rperm

    @ staticmethod
    def elem_size(vert):
        """Calculates the length, area, and volume of a line, triangle, and tetrahedral reference elements given
        the coordinates of their vertices.
        """
        nvert = vert.shape[0]
        ncoord = vert.shape[1]
        dim = nvert - 1
        elem_size = 0

        if dim == 1:
            elem_size = np.linalg.norm(vert[1,:] - vert[0,:])
        elif dim == 2:
            if ncoord == 2:
                vert_zero = np.zeros((nvert, ncoord+1))
                vert_zero[:, :-1] = vert
                vert = vert_zero
            elem_size = 1/2 * np.linalg.norm(np.cross(vert[1,:]-vert[0,:], vert[2, :]-vert[1,:]))
        elif dim == 3:
            elem_size = 1/6 * np.linalg.norm(np.dot(np.cross(vert[1,:]-vert[0,:], vert[2, :]-vert[1,:]), vert[3, :]-vert[0, :]))

        return elem_size

    @ staticmethod
    def face_to_vert(dim):
        """Returns the facet to vertex connectivity. It assumes that the node 1 is at [-1,-1] and the rest are numbered
        moving in counter clockwise direction. The faces are numbered with the node number oppositeto them.
        E.g., The slanted facet of the triangle reference element is face 1.
        """
        f2v = 0
        if dim == 1:
            f2v = np.array([0, 1])
        elif dim ==2:
            # f2v = np.array([[1, 2, 0], [2, 0, 1]])
            # to change facet numbering as in Hesthaven and Warburton's book uncomment the line below
            f2v = np.array([[0, 1, 2], [1, 2, 0]])
        if f2v is 0:
            raise ValueError("Dimension entered not implemented.")

        return f2v

    @staticmethod
    def normals(vert):
        """Calculates the surface normals of the reference element given the vertices."""
        nvert = vert.shape[0]
        ncoord = vert.shape[1]
        dim = nvert - 1
        vz = np.array([0, 0, 1])    # normal vector out of the page
        f2v = Ref2D_SBP.face_to_vert(dim)
        vn = np.zeros((nvert, 3))
        vn_nor = np.zeros((nvert, 3))

        for i in range(0, dim+1):
            vf = f2v[:, i]      # vertices on the face
            coord = vert[vf, :]
            v1 = 0
            if ncoord < 3:
                v1_zero = np.zeros((1, ncoord + 1))
                v1_zero[:, :-1] = coord[1,:] - coord[0,:]
                v1 = v1_zero
            else:
                v1 = coord[1,:] - coord[0,:]

            vn[i,:] = np.cross(v1, vz)                              # calculate the normal vectors
            vn_nor[i,:] = vn[i,:]/(np.sqrt(np.sum(vn[i,:]**2)))     # normalize the normal vectors

        return {'vn': vn, 'vn_nor': vn_nor}

    @ staticmethod
    def fmask_2d(r, s, vert, rsf=None):
        """Gets the node number (row in r) of the nodes on the facets. If quadrature nodes on the facets are provided,
        those volume nodes that correspond to the quadrature nodes are also returned. An assumption that the
        reference element is right triangle is made.
        Args:
            r - x coordinate of the volume nodes on the reference element
            s - y coordinate of the volume nodes on the reference element
            vert - (3 X 2) array containing the coordinates of the vertices of the reference element
            rsf -  (3 X nq X 2) array containing the coordinates of the facet quadrature nodes on the 3 facets
        Returns:
            fmask - (nn X 3) array containing the row numbers of the volume nodes on the facets of the reference
                    triangle; nn is the number of volume nodes per facet
            fmask_q - (qq X 3) array containing the row numbers of the volume nodes that are on the same location on
                    the facet as the quadrature nodes (required to construct SBP-diagE type operators)
            """

        tol = 1e-10
        fmask_q = None

        # fmask1_unsorted = ((np.abs(r + s - (vert[1,0]+vert[1,1])) < tol).nonzero())[0]       # nodes on face 1
        # fmask2_unsorted = ((np.abs(r - vert[0, 0]) < tol).nonzero())[0]                      # nodes on face 2
        # fmask3_unsorted = ((np.abs(s - vert[0, 1]) < tol).nonzero())[0]                      # nodes on face 3
        fmask2_unsorted = ((np.abs(r + s - (vert[1, 0] + vert[1, 1])) < tol).nonzero())[0]  # nodes on face 1
        fmask3_unsorted = ((np.abs(r - vert[0, 0]) < tol).nonzero())[0]  # nodes on face 2
        fmask1_unsorted = ((np.abs(s - vert[0, 1]) < tol).nonzero())[0]  # nodes on face 3

        # sort to read nodes counterclockwise on facets
        # fmask1 = fmask1_unsorted[np.argsort(s[fmask1_unsorted].flatten())]
        # fmask2 = fmask2_unsorted[np.argsort(-s[fmask2_unsorted].flatten())]
        # fmask3 = fmask3_unsorted[np.argsort(r[fmask3_unsorted].flatten())]
        fmask1 = fmask1_unsorted[np.argsort(r[fmask1_unsorted].flatten())]
        fmask2 = fmask2_unsorted[np.argsort(s[fmask2_unsorted].flatten())]
        fmask3 = fmask3_unsorted[np.argsort(-s[fmask3_unsorted].flatten())]
        fmask = np.array([fmask1, fmask2, fmask3]).T

        if rsf is not None:
            # check if the location of the quadrature nodes and the volume nodes coincide on the facets
            fmask1_q = fmask1[np.abs(r[fmask1].flatten() - rsf[0][:,0]) + np.abs(s[fmask1].flatten() - rsf[0][:,1]) < tol]
            fmask2_q = fmask2[np.abs(r[fmask2].flatten() - rsf[1][:,0]) + np.abs(s[fmask2].flatten() - rsf[1][:,1]) < tol]
            fmask3_q = fmask3[np.abs(r[fmask3].flatten() - rsf[2][:,0]) + np.abs(s[fmask3].flatten() - rsf[2][:,1]) < tol]
            fmask_q = np.array([fmask1_q, fmask2_q, fmask3_q]).T

        return {'fmask': fmask, 'fmask_q': fmask_q}

    @ staticmethod
    def make_r(V, Vf, sbp_family="gamma", fmask=None, fmask_q=None):
        """Builds the interpolation/extrapolation matrix for all facets.
        Args:
            V - (nnodes X ns) array of the volume Vandermonde matrix (where nnodes are the number of volume nodes and
                ns is the number of shape functions; ns=(p+1)(p+2)/2
            Vf - a list of (nq X ns) arrays of the facet Vandermonde matrices ordered as [Vf1, Vf2, Vf3], nq is the
                number of quadrature nodes
            sbp_family - "gamma" , "omega", or "diage"
            fmask - (nn X 3) array containing the row numbers of the volume nodes on the facets of the reference
                    triangle; nn is the number of volume nodes per facet
            fmask_q - (qq X 3) array containing the row numbers of the volume nodes that are on the same location on
                    the facet as the quadrature nodes (required to construct SBP-diagE type operators)
        Returns:
            R1 - (nq X ns) array, the extrapolation matrix for face 1
            R2 - (nq X ns) array, the extrapolation matrix for face 2
            R3 - (nq X ns) array, the extrapolation matrix for face 3
        """
        sbp_family = str.lower(sbp_family)
        R1 = np.zeros((Vf[0].shape[0], V.shape[0]))
        R2 = np.zeros((Vf[1].shape[0], V.shape[0]))
        R3 = np.zeros((Vf[2].shape[0], V.shape[0]))

        if sbp_family == "gamma" and (len(fmask_q[:, 0]) != len(fmask[:,0])):
            fmask1 = fmask[:, 0]
            fmask2 = fmask[:, 1]
            fmask3 = fmask[:, 2]
            # R on face 1
            R_temp1 = Vf[0] @ np.linalg.pinv(V[fmask1, :])
            R1[:, fmask1] = R_temp1
            # R on face 2
            R_temp2 = Vf[1] @ np.linalg.pinv(V[fmask2, :])
            R2[:, fmask2] = R_temp2
            # R on face 3
            R_temp3 = Vf[2] @ np.linalg.pinv(V[fmask3, :])
            R3[:, fmask3] = R_temp3

        elif sbp_family == "diage" or (sbp_family=="gamma" and (len(fmask_q[:, 0]) == len(fmask[:,0]))):
            fmask1_q = fmask_q[:, 0]
            fmask2_q = fmask_q[:, 1]
            fmask3_q = fmask_q[:, 2]

            for i in range(0, len(fmask1_q)):
                # R on face 1
                R1[i, fmask1_q[i]] = 1
                # R on face 2
                R2[i, fmask2_q[i]] = 1
                # R on face 3
                R3[i, fmask3_q[i]] = 1

        else:
            # R on face 1
            R1 = Vf[0] @ np.linalg.pinv(V)
            # R on face 2
            R2 = Vf[1] @ np.linalg.pinv(V)
            # R on face 3
            R3 = Vf[2] @ np.linalg.pinv(V)

        return {'R1': R1, 'R2': R2, 'R3': R3}

    @staticmethod
    def nodes_sbp_2d(p, sbp_family):
        """Returns the nodal locations on the reference element"""
        # set dimension
        dim = 2
        # get the vertices of the reference element
        vert = Ref2D_SBP.simplex_vertices(dim)
        # get cubature data on the a reference element
        cub_data = CubatureRules.cub_tri_volume(p, sbp_family)
        cub = SimpleNamespace(**cub_data)
        nnodes = len(cub.r)

        # get Barycentric coordinat of the cubature nodes
        b = Ref2D_SBP.cartesian_to_barycentric2D(cub.r, cub.s, cub.cub_vert)

        # get the coordinates of the cubature nodes on the reference triangle
        # (if it is different form the reference element for which the cubature rule is given for)
        rs_data = Ref2D_SBP.barycentric_to_cartesian(b, vert)
        r = rs_data[:, 0].reshape(len(rs_data[:, 0]), 1)
        s = rs_data[:, 1].reshape(len(rs_data[:, 1]), 1)

        return {'r': r, 's': s}

    @staticmethod
    def make_sbp_operators2D(p, sbp_family="gamma"):

        sbp_family = str.lower(sbp_family)
        dim = 2
        nface = dim + 1
        ns = (p+1)*(p+2)/2
        tol = 1e-10
        vert = Ref2D_SBP.simplex_vertices(dim)

        # get the cubature points
        cub_data = CubatureRules.cub_tri_volume(p, sbp_family)
        cub = SimpleNamespace(**cub_data)
        nnodes = len(cub.r)

        # get Barycentric coordinate of the cubature nodes
        b = Ref2D_SBP.cartesian_to_barycentric2D(cub.r, cub.s, cub.cub_vert)

        # get the coordinates of the cubature nodes on the reference triangle
        # (if it is different form the reference element for which the cubature rule is given for)
        rs_data = Ref2D_SBP.barycentric_to_cartesian(b, vert)
        r = rs_data[:, 0].reshape(len(rs_data[:,0]), 1)
        s = rs_data[:, 1].reshape(len(rs_data[:,1]), 1)

        # identrify the nodal symmetry (with respect to x=y line) and symmetry group (S3, S211, and S111)
        nodal_sym = Ref2D_SBP.nodal_sym_map2D(r, s)
        # sym_grps = Ref2D_SBP.sym_group_map2D(r, s)
        # sym_grp = sym_grps['sym_grp']

        # get the cubature rule for the facets and find the symmetry group of the facet quadrature nodes
        xqf, wqf = CubatureRules.quad_line_volume(p, "LGL")
        sym_grps_xqf = Ref2D_SBP.sym_group_map2D(xqf)
        sym_grp_xqf = sym_grps_xqf['sym_grp']

        # get the barycentric coordinate for the facet quadrature nodes
        bf = Ref2D_SBP.cartesian_to_barycentric2D(xqf)

        # get the coordinates of the quadrature points on the facets
        # rsf1 = Ref2D_SBP.barycentric_to_cartesian(bf, np.array([vert[1, :], vert[2, :]]))   # facet 1
        # rsf2 = Ref2D_SBP.barycentric_to_cartesian(bf, np.array([vert[2, :], vert[0, :]]))   # facet 2
        # rsf3 = Ref2D_SBP.barycentric_to_cartesian(bf, np.array([vert[0, :], vert[1, :]]))   # facet 3
        rsf1 = Ref2D_SBP.barycentric_to_cartesian(bf, np.array([vert[0, :], vert[1, :]]))  # facet 1
        rsf2 = Ref2D_SBP.barycentric_to_cartesian(bf, np.array([vert[1, :], vert[2, :]]))  # facet 2
        rsf3 = Ref2D_SBP.barycentric_to_cartesian(bf, np.array([vert[2, :], vert[0, :]]))  # facet 3

        rsf = np.array([rsf1, rsf2, rsf3])

        # calculate the Vandermonde matrix for the facet quadrature nodes (on face 1 - the slant)
        Vf = []
        Vf.append(Ref2D_DG.vandermonde_2d(p, rsf1[:, 0], rsf1[:, 1]))
        Vf.append(Ref2D_DG.vandermonde_2d(p, rsf2[:, 0], rsf2[:, 1]))
        Vf.append(Ref2D_DG.vandermonde_2d(p, rsf3[:, 0], rsf3[:, 1]))
        # Vdxf = Ref2D_DG.grad_vandermonde2d(p, rsf1[:, 0], rsf1[:, 1])

        # calculate the Vandermonde matrix at the cubature nodes
        V = Ref2D_DG.vandermonde_2d(p, r, s)

        # calculate the derivative of the Vandermonde matrix at the cubature nodes
        V_der = Ref2D_DG.grad_vandermonde2d(p, r, s)
        Vdr = V_der['vdr']
        Vds = V_der['vds']

        # get permutation matrix to go from x to y (e.g., obtain Es from Er)
        I = np.eye(nnodes)
        Pv = np.zeros((dim, nnodes, nnodes))
        for i in range(0, dim):
            Pv[i, :, :] = I[nodal_sym[i, :]-1, :]

        # get H: the volume norm matrix
        elem_size = Ref2D_SBP.elem_size(vert)
        H = np.diag((cub.w).flatten()) * (elem_size/np.sum(cub.w))

        # get B: the facet norm matrix
        B = np.diag(wqf.flatten())      # unscaled B matrix (i.e., on the 1D reference element [-1, 1])
        B_list =[]                      # containes B scaled by the size of the facets of the reference element
        for i in range(0, dim+1):
            # vertf = np.roll(vert, i)[1:3, :]    # get the vertices of the facets i
            vertf = np.roll(vert, i)[0:2, :]
            elem_sizef = Ref2D_SBP.elem_size(vertf)
            B_list.append(np.diag(wqf.flatten()) * (elem_sizef/np.sum(wqf)))

        # get N: the surface normal vectors
        normals_data = Ref2D_SBP.normals(vert)
        vn = normals_data['vn_nor']  # normalized normal vector
        nx = vn[:, 0]
        ny = vn[:, 1]

        # get R: the interpolation/extrapolation matrix on face 1
        if sbp_family != "omega":
            fmask_data = Ref2D_SBP.fmask_2d(r, s, vert, rsf)
        else:
            fmask_data = Ref2D_SBP.fmask_2d(r, s, vert)
        fmask = fmask_data['fmask']
        fmask_q = fmask_data['fmask_q']
        R_data = Ref2D_SBP.make_r(V, Vf, sbp_family, fmask, fmask_q)
        R = SimpleNamespace(**R_data)
        R_list = [R.R1, R.R2, R.R3]

        # get Er: the surface integral matrix in the x-direction
        Er = np.zeros((nnodes, nnodes))

        for i in range(0, nface):
            Er = Er + nx[i] * (R_list[i].T@ B_list[i] @ R_list[i])

        # get Dr: the derivative operator of the x-direction
        more_nodes = int(nnodes - ns)
        if more_nodes > 0:
            null_V = null_space(V.T)
            W = null_V
            V_tilde = np.block([V, W])
            zz = np.zeros((more_nodes, more_nodes))
            Wx = (np.linalg.inv(V_tilde.T @ H)) @ ((1/2*V_tilde.T @ Er @ W)
                 + np.block([[-Vdr.T @ H @ W + (1/2*V.T @ Er @ W)], [zz]]))

            Vx_tilde = np.block([Vdr, Wx])
            Dr = Vx_tilde @ np.linalg.inv(V_tilde)
        else:
            Dr = Vdr @ np.linalg.inv(V)

        # get Qr: the weak derivative operator
        Qr = H @ Dr

        # permute directional operators to the y axis
        Es = Pv[1, :, :] @ Er @ Pv[1, :, :]
        Qs = Pv[1, :, :] @ Qr @ Pv[1, :, :]
        Ds = Pv[1, :, :] @ Dr @ Pv[1, :, :]

        return {'H': H, 'B': B, 'Dr': Dr, 'Ds': Ds, 'Er': Er, 'Es': Es, 'Qr': Qr, 'Qs': Qs, 'B1': B_list[0],
                'B2': B_list[1], 'B3': B_list[2], 'R1': R.R1, 'R2': R.R2, 'R3': R.R3,'vert': cub.cub_vert,
                'r': r, 's': s, 'rsf': rsf, 'V': V, 'Vf': Vf, 'bary': b, 'baryf': bf, 'nx': nx, 'ny': ny}


#M = Ref2D_DG.mass_matrix(1)


# print(M)
# p = 3
# n = int((p+1)*(p+2)/2)
# x_ref, y_ref = Ref2D_DG.nodes_2d(p)
#
# r, s = Ref2D_DG.xytors(x_ref, y_ref)

# edge_nodes = Ref2D_DG.fmask_2d(r, s, x_ref, y_ref)
# fmask = edge_nodes['fmask']
#
#
# v = Ref2D_DG.vandermonde_2d(p, r, s)
#
# drvtv = Ref2D_DG.derivative_2d(p, r, s, v)
# Dr = drvtv['Dr']
# Ds = drvtv['Ds']
#
#
# lift = Ref2D_DG.lift_2d(p, r, s, fmask)
# w, r, s= Ref2D_DG.quad_rule_tri(p, 'Liu-Vinokur')
# shp_data = Ref2D_SBP.shape_tri(p, sbp_family)

# p = 4
# sbp_family = "diagE"
# shp_data = Ref2D_SBP.make_sbp_operators2D(p, sbp_family)

# shp = shp_data['shp']
# shpx = shp_data['shpx']
# shp_data