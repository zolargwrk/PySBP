import numpy as np
from matplotlib.tri.triangulation import Triangulation
from mesh.mesh_tools import MeshTools2D


def mytriplot(ax, *args, **kwargs):
    """
    Draw a unstructured triangular grid as lines and/or markers.
    The triangulation to plot can be specified in one of two ways; either::
      triplot(triangulation, ...)
    where triangulation is a `.Triangulation` object, or
    ::
      triplot(x, y, ...)
      triplot(x, y, triangles, ...)
      triplot(x, y, triangles=triangles, ...)
      triplot(x, y, mask=mask, ...)
      triplot(x, y, triangles, mask=mask, ...)
    in which case a Triangulation object will be created.  See `.Triangulation`
    for a explanation of these possibilities.
    The remaining args and kwargs are the same as for `~.Axes.plot`.
    Returns
    -------
    lines : `~matplotlib.lines.Line2D`
        The drawn triangles edges.
    markers : `~matplotlib.lines.Line2D`
        The drawn marker nodes.
    """
    import matplotlib.axes

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    x, y, edges = (tri.x, tri.y, tri.edges)
    # get mesh length in the x and y directions
    r = args[1]
    s = args[2]
    etov = args[3]
    p_map = args[4]
    Lx = args[5]
    Ly = args[6]

    # Decode plot format string, e.g., 'ro-'
    fmt = args[0] if args else ""
    linestyle, marker, color = matplotlib.axes._base._process_plot_format(fmt)

    # Insert plot format string into a copy of kwargs (kwargs values prevail).
    kw = kwargs.copy()
    for key, val in zip(('linestyle', 'marker', 'color'),
                        (linestyle, marker, color)):
        if val is not None:
            kw[key] = kwargs.get(key, val)

    curve_mesh = kw['curve_mesh']

    # Draw lines without markers.
    # Note 1: If we drew markers here, most markers would be drawn more than
    #         once as they belong to several edges.
    # Note 2: We insert nan values in the flattened edges arrays rather than
    #         plotting directly (triang.x[edges].T, triang.y[edges].T)
    #         as it considerably speeds-up code execution.
    linestyle = kw['linestyle']
    kw_lines = {
        'lw': kw['lw'],
        'linestyle': kw['linestyle'],
        'marker': 'None',  # No marker to draw.
        'zorder': kw.get('zorder', 1),  # Path default zorder is used.
    }

    if linestyle not in [None, 'None', '', ' ']:
        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)

        tri_lines_x2, tri_lines_y2 = add_points(r, s, tri_lines_x, tri_lines_y, x, y, etov, p_map=p_map, Lx=Lx, Ly=Ly,
                                                curve_mesh=curve_mesh)

        tri_lines = ax.plot(tri_lines_x2.ravel(), tri_lines_y2.ravel(), **kw_lines)
    else:
        tri_lines = ax.plot([], [], **kw_lines)

    # Draw markers separately.
    marker = kw['marker']
    kw_markers = {
        'lw': kw['lw'],
        'marker': kw['marker'],
        'color': kw['color'],
        'linestyle': 'None',  # No line to draw.
    }
    if marker not in [None, 'None', '', ' ']:
        tri_markers = ax.plot(x, y, **kw_markers)
    else:
        tri_markers = ax.plot([], [], **kw_markers)

    return tri_lines + tri_markers

def add_points(r, s, tri_lines_x, tri_lines_y, vx, vy, etov, p_map=2, Lx=1, Ly=1, curve_mesh=True):
    """ Adds more nodes between the vertices of the triangles and applies mesh curvature
    :arg tri_line_x - n X 3 matrix where the first two columns contain the x coordinates of connected vertices
    :arg tri_line_y - n X 3 matrix where the first two columns contain the y coordinates of connected vertices"""

    tri_lines_x2 = np.zeros([1, 3])
    tri_lines_y2 = np.zeros([1, 3])

    n = 100    # number of points between vertices
    for j in range(tri_lines_x.shape[0]):
        # get the slope of an edge
        dx = (tri_lines_x[j, 1] - tri_lines_x[j, 0])
        if np.abs(dx) >= 1e-10:
            a = (tri_lines_y[j, 1] - tri_lines_y[j, 0])/dx
            # add more points between the two vertices
            x = np.linspace(tri_lines_x[j, 0], tri_lines_x[j, 1], n)
            y = tri_lines_y[j, 0] + a * (x - tri_lines_x[j, 0])
        else:
            # add more points between the two vertices
            x = np.linspace(tri_lines_x[j, 0], tri_lines_x[j, 1], n)
            y = np.linspace(tri_lines_y[j, 0], tri_lines_y[j, 1], n)

        # apply mesh curvature
        # etov_elem = etov[j].reshape((1, -1))
        # vx_elem = vx[etov_elem].reshape((-1, 1))
        # vy_elem = vy[etov_elem].reshape((-1, 1))
        curved_data = MeshTools2D.curve_mesh2d(r, s, x.reshape((-1, 1)), y.reshape(-1, 1), vx, vy, etov, p_map, Lx, Ly,
                                               curve_mesh=curve_mesh)
        x = curved_data['x']
        y = curved_data['y']

        # add nodes to tri_lines
        xnew = np.insert(x.reshape(-1, 2), 2, np.nan, axis=1)
        ynew = np.insert(y.reshape(-1, 2), 2, np.nan, axis=1)
        tri_lines_x2 = np.vstack([tri_lines_x2, xnew])
        tri_lines_y2 = np.vstack([tri_lines_y2, ynew])

    # delete first rows because they are zero
    tri_lines_x2 = np.delete(tri_lines_x2, (0), axis=0)
    tri_lines_y2 = np.delete(tri_lines_y2, (0), axis=0)

    return tri_lines_x2, tri_lines_y2

