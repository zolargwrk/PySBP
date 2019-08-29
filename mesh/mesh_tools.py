import meshzoo
import meshio
import matplotlib.pyplot as plt
import numpy as np


points, cells = meshzoo.line(8)
meshio.write_points_cells('rectangle.e', points, {'triangle': cells})

# print(meshio.rectangle.points)
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))
plt.plot(x, np.sin(x))
plt.show()