import numpy as np
from scipy import sparse

# define problem input
def poisson1D_problem_input (x=None, xl=None, xr=None, n=None):
    def var_coef (n):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        b = 1
        return b

    def exact_solution(x):
        u_exact = np.cos(30*x)
        return u_exact

    def boundary_conditions(xl, xr):
        uD_left = np.cos(30 * xl)  # None      # Dirichlet boundary at the left boundary
        uD_right = np.cos(30 * xr)  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
        uN_left = None  # -30*np.sin(30*xl) #None     # Neumann boundary at the left boundary
        uN_right = None  # -30*np.sin(30*xr)  # None    # Neumann boundary at the right boundary
        return {'uD_left': uD_left, 'uD_right': uD_right, 'uN_left': uN_left, 'uN_right': uN_right}

    def source_term(x):
        f = 900 * np.cos(30 * x)
        return f

    def adjoint_source_term(x):
        g = np.cos(30*x)
        return g

    def adjoint_bndry (xl, xr):
        psiD_left = np.cos(30 * xl)  # None      # Dirichlet boundary at the left boundary
        psiD_right = np.cos(30 * xr)  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
        psiN_left = None  # -30*np.sin(30*xr) #None     # Neumann boundary at the left boundary
        psiN_right = None  # -30*np.sin(30*xl)  # None    # Neumann boundary at the right boundary
        return {'psiD_left': psiD_left, 'psiD_right': psiD_right, 'psiN_left': psiN_left, 'psiN_right': psiN_right}

    def exact_adjoint(x):
        psi = 1/900*np.cos(30*x) + (1-1/900)*(np.cos(30)-1)*x + (1-1/900)
        return psi

    def exact_functional(xl, xr):
        J_exact = 1/30*np.sin(30*xr) - 1/30*np.sin(30*xl)
        return J_exact

    def calc_functional(u, h_mat, rx):
        rx_global = np.diag(1 / rx[0, :], 0)  # geometric factor rx = 1/jac
        h_mat_global = sparse.block_diag([h_mat])  # concatenate norm matrix to form global
        rh = sparse.kron(rx_global, h_mat_global)
        J = (np.ones((1, rh.shape[0])) @ rh @ u)[0][0]
        return J

    def choose_output():
        """Choose which output to see
        Input:  primal: output results of primal problem
                adjoint: output results of adjoint problem
                functional: output results of functional convergence
                all: outputs all three of the above"""
        outs = 'primal'
        # outs = 'adjoint'
        # outs = 'functional'
        # outs = 'all'
        return outs

    return {'var_coef': var_coef, 'exact_solution': exact_solution, 'boundary_conditions': boundary_conditions,
            'source_term': source_term, 'adjoint_source_term':adjoint_source_term, 'adjoint_bndry':adjoint_bndry,
            'exact_adjoint':exact_adjoint, 'exact_functional': exact_functional, 'calc_functional':calc_functional,
            'choose_output': choose_output}