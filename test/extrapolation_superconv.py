import numpy as np
from mesh.mesh_generator import MeshGenerator1D, MeshGenerator2D
from mesh.mesh_tools import MeshTools1D, MeshTools2D
from src.assembler import Assembler
from types import SimpleNamespace
from src.error_conv import calc_conv
from solver.plot_figure import plot_figure_1d, plot_figure_2d, plot_conv_fig


def extrpolation_superconv_1D(p, xl, xr, nelem, quad_type, b=1, app=1, n=1, nrefine=4, refine_type='elem'):

    # get the mesh and apply refinement
    self_assembler = Assembler(p, quad_type, boundary_type=None)
    rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a=1, nelem=1, n=n, b=b, app=app)

    errs = list()
    nelems = list()
    dofs = list()
    for i in range(0, nrefine):
        if i == 0:
            mesh = MeshGenerator1D.line_mesh(p, xl, xr, n, nelem, quad_type, b, app)
        else:
            if refine_type == 'trad':
                mesh = MeshTools1D.trad_refine_uniform_1d(rhs_data, p, quad_type, b, app)
                n = mesh['n']
            else:
                mesh = MeshTools1D.hrefine_uniform_1d(rhs_data)

        # get nodes and extrapolation/interpolation operator
        nelem = mesh['nelem']
        rhs_data = Assembler.assembler_1d(self_assembler, xl, xr, a=1, nelem=nelem, n=n, b=b, app=app)
        rdata = SimpleNamespace(**rhs_data)
        n = rdata.n
        x = rdata.x.reshape((n, nelem), order='F')
        tl = rdata.tl
        tr = rdata.tr

        dofs.append(n * nelem)
        nelems.append(nelem)

        # specify exact solution
        uexact = lambda x: x**5
        ue_xl = uexact(xl)
        ue_xr = uexact(xr)

        # extrapolate the exact solution to the boundaries
        ue = uexact(x)
        u_xl = (tl.T @ (ue[:, 0].reshape((-1, 1))))[0][0]
        u_xr = (tr.T @ (ue[:, -1].reshape((-1, 1))))[0][0]

        # calculate error
        err_uxl = np.abs(ue_xl - u_xl)
        err_uxr = np.abs(ue_xr - u_xr)
        errs.append(err_uxr)

    conv_start = 3
    conv_end = nrefine - 0
    if refine_type == 'trad':
        hs = (xr - xl) / (np.asarray(dofs))
    else:
        hs = (xr - xl) / (np.asarray(nelems))
    conv_rate = calc_conv(hs, errs, conv_start, conv_end)
    print(np.asarray(conv_rate))
    plot_conv_fig(hs, errs, conv_start, conv_end)
    plot_figure_1d(x, ue)

    return conv_rate


conv_rates = extrpolation_superconv_1D(p=2, xl=-1, xr=1, nelem=1, quad_type='LG', n=1, nrefine=6, refine_type='elem')
