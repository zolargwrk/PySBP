import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from mesh.mesh_generator import MeshGenerator2D

class MeshPlot:

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

mesh = MeshGenerator2D.rectangle_mesh(0.5)
# mesh = MeshGenerator2D.triangle_mesh(11)
fig = MeshPlot.mesh_plot_2d('rec.vtu')