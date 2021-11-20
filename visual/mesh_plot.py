import numpy as np
import matplotlib.pyplot as plt
#import vtk
#from vtk.util.numpy_support import vtk_to_numpy
from mesh.mesh_generator import MeshGenerator2D
from mesh.mesh_tools import MeshTools2D
from src.ref_elem import Ref2D_SBP
from types import SimpleNamespace
from visual.mytriplot import mytriplot


class MeshPlot:
    """
    @staticmethod
    def mesh_plot_2d(file_vtu):

        # read data
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_vtu)
        reader.Update()
        data = reader.GetOutput()

        # get mesh data
        points = data.GetPoints()
        npts = points.GetNumberOfPoints()
        x = vtk_to_numpy(points.GetData())

        triangles = vtk_to_numpy(data.GetCells().GetData())
        ntri = triangles.size // 4
        tri = np.take(triangles, [n for n in range(triangles.size) if n % 4 != 0]).reshape(ntri, 3)
        # u = vtk_to_numpy(data.GetPointData().GetArray(2))

        # plot
        plt.figure(figsize=(8, 8))
        plt.triplot(x[:, 0], x[:, 1], tri)
        plt.gca().set_aspect('equal')
        plt.show()

        return
    """
    @staticmethod
    def plot_mesh_2d(nelem, r, s, x, y, xf, yf, vx, vy, etov, p_map=2, Lx=1, Ly=1, showFacetNodes=False,
                     showVolumeNodes=False, saveMeshPlot=False, curve_mesh=True, sbp_family=''):

        # xc, yc = MeshTools2D.curve_mesh2d(x, y, Lx=Lx, Ly=Ly, func=None)
        # vx, vy = MeshTools2D.curve_mesh2d(vx, vy, Lx=Lx, Ly=Ly, func=None)
        # xfc, yfc = MeshTools2D.curve_mesh2d(xf, yf, Lx=Lx, Ly=Ly, func=None)

        # x, y, xf, yf are already curved in the assembler
        triangles = etov.tolist()
        plt.plot()
        ax = plt.gca()
        mytriplot(ax, vx, vy, triangles, 'b-', r, s, etov, p_map, Lx, Ly, curve_mesh=curve_mesh, lw=1, linestyle='-')
        if showFacetNodes:
            plt.scatter(xf, yf, marker='s', c='w', s=12, edgecolors='r', linewidths=1)
        if showVolumeNodes:
            plt.scatter(x, y, marker='o', c='k', s=10)
        ax.set_aspect('equal')

        if saveMeshPlot:
            path = 'C:\\Users\\Zelalem\\OneDrive - University of Toronto\\UTIAS\\Research\\PySBP\\visual\\poisson2d_results\\mesh\\'
            plt.savefig(path + 'nelem_{}_pmap_{}_{}.pdf'.format(nelem, p_map, sbp_family), format='pdf')

        plt.show()

        return


# mesh = MeshGenerator2D.rectangle_mesh(0.5, 0, 1, 0, 1)
# geo = mesh['geo']
# mesh = MeshGenerator2D.triangle_mesh(11)
# fig = MeshPlot.mesh_plot_2d('square_4elem.vtu')
# fig = MeshPlot.mesh_plot_2d(geo)

#---------------------------
# # set operator
# sbp_family = 'gamma'
# p = 2
#
# # the rectangular domain
# h = 1
# bL = 0
# bR = 20
# bB = -5
# bT = 5
# Lx = np.abs(bR-bL)
# Ly = np.abs(bT-bB)
# # obtain data on the reference element
# sbp_ref_data = Ref2D_SBP.make_sbp_operators2D(p, sbp_family)
# sbpref = SimpleNamespace(**sbp_ref_data)
# r = sbpref.r
# s = sbpref.s
# # generate mesh
# mesh = MeshGenerator2D.rectangle_mesh(h, bL, bR, bB, bT)
# vx = mesh['vx']
# vy = mesh['vy']
# etov = mesh['etov']
# # apply affine mapping and obtain mesh location of all nodes on the physical element
# x, y = MeshTools2D.affine_map_2d(vx, vy, r, s, etov)
# rf = sbpref.rsf[2, :, 0]
# sf = sbpref.rsf[2, :, 1]
# baryf = sbpref.baryf
# xf, yf = MeshTools2D.affine_map_facet_sbp_2d(vx, vy, rf, sf, etov, baryf)
#
# MeshPlot.plot_mesh_2d(x, y, xf, yf, vx, vy, etov, Lx, Ly, showFacetNodes=False, showVolumeNodes=True)