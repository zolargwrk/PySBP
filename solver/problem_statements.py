import numpy as np
from scipy import sparse

# define problem input
def poisson1D_problem_input (x=None, xl=None, xr=None, n=None):
    def choose_output():
        """Choose which output to see
        Input:  primal: output results of primal problem
                adjoint: output results of adjoint problem
                functional: output results of functional convergence
                all: outputs all three of the above"""
        prob = 'primal'
        # prob = 'adjoint'
        # prob = 'all'
        func_conv = 0
        plot_sol = 1
        plot_err = 0
        show_eig = 1
        return {'prob': prob, 'func_conv': func_conv, 'plot_sol': plot_sol, 'plot_err': plot_err, 'show_eig': show_eig}

    def var_coef (x=1):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        b = 1  # x**0
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
        b = var_coef(x.shape[0])
        f = b*900 * np.cos(30 * x)
        return f

    def adjoint_source_term(x):
        g = np.cos(30*x)
        return g

    def adjoint_bndry (xl, xr):
        b = var_coef()
        psiD_left = 0       #1/b*np.cos(30 * xl)  # None      # Dirichlet boundary at the left boundary
        psiD_right = 0      #1/b*np.cos(30 * xr)  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
        psiN_left = None    #-30*np.sin(30*xl) #None     # Neumann boundary at the left boundary
        psiN_right = None   #-30*np.sin(30*xr)  # None    # Neumann boundary at the right boundary
        return {'psiD_left': psiD_left, 'psiD_right': psiD_right, 'psiN_left': psiN_left, 'psiN_right': psiN_right}

    def exact_adjoint(x):
        b = var_coef()
        # psi = 1/900*np.cos(30*x) + (1-1/900)*(np.cos(30)-1)*x + (1-1/900)
        psi = 1 / (900*b) * np.cos(30 * x) + x/(900*b) *(1 - np.cos(30)) - 1 / (900*b)
        return psi

    def exact_functional(xl, xr):
        J_exact = (xr/2 + 1/120 * np.sin(60*xr)) - (xl/2 + 1/120 * np.sin(60*xl))
        return J_exact

    def calc_functional(u, g, h_mat, rx):
        rx_global = np.diag(1 / rx[0, :], 0)  # geometric factor rx = 1/jac
        h_mat_global = sparse.block_diag([h_mat])  # concatenate norm matrix to form global
        rh = sparse.kron(rx_global, h_mat_global)
        J = (np.ones((1, rh.shape[0])) @ rh @ (g * u))[0][0]
        return J

    return {'var_coef': var_coef, 'exact_solution': exact_solution, 'boundary_conditions': boundary_conditions,
            'source_term': source_term, 'adjoint_source_term':adjoint_source_term, 'adjoint_bndry':adjoint_bndry,
            'exact_adjoint':exact_adjoint, 'exact_functional': exact_functional, 'calc_functional':calc_functional,
            'choose_output': choose_output}

def advection1D_problem_input (x=None, xl=None, xr=None, n=None):
    def choose_output():
        """Choose which output to see
        Input:  primal: output results of primal problem
                adjoint: output results of adjoint problem
                functional: output results of functional convergence
                all: outputs all three of the above"""
        prob = 'primal'
        # prob = 'adjoint'
        # prob = 'all'
        func_conv = 1
        plot_sol = 0
        plot_err = 1
        show_eig = 0
        return {'prob': prob, 'func_conv': func_conv, 'plot_sol': plot_sol, 'plot_err': plot_err, 'show_eig': show_eig}

    def var_coef (x=1):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        a = 1  # x**0
        return a

    def exact_solution(x):
        u_exact = np.cos(30*x)
        return u_exact

    def boundary_conditions(xl, xr):
        uD_left = np.cos(30 * xl)  # None      # Dirichlet boundary at the left boundary
        uD_right = None  #np.cos(30 * xr)  # None     # Dirichlet boundary at the right boundary

        return {'uD_left': uD_left, 'uD_right': uD_right}

    def source_term(x):
        a = var_coef(x.shape[0])
        f = - a*30 * np.sin(30 * x)
        return f

    def adjoint_source_term(x):
        g = np.cos(30*x)
        return g

    def adjoint_bndry (xl, xr):
        a = var_coef()
        psiD_left = None    # np.cos(30 * xl)/a  # None      # Dirichlet boundary at the left boundary
        psiD_right = np.cos(30 * xr)/a  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary

        return {'psiD_left': psiD_left, 'psiD_right': psiD_right}

    def exact_adjoint(x):
        a = var_coef()
        psi = 1/a * 1/30*np.sin(30*x) + np.cos(30) - 1/30*np.sin(30)
        return psi

    def exact_functional(xl, xr):
        J_exact = (xr/2 + 1/120 * np.sin(60*xr)) - (xl/2 + 1/120 * np.sin(60*xl))
        return J_exact

    def calc_functional(u, g,  h_mat, rx):
        rx_global = np.diag(1 / rx[0, :], 0)  # geometric factor rx = 1/jac
        h_mat_global = sparse.block_diag([h_mat])  # concatenate norm matrix to form global
        rh = sparse.kron(rx_global, h_mat_global)
        J = (np.ones((1, rh.shape[0])) @ rh @ (g * u))[0][0]
        return J

    return {'var_coef': var_coef, 'exact_solution': exact_solution, 'boundary_conditions': boundary_conditions,
            'source_term': source_term, 'adjoint_source_term':adjoint_source_term, 'adjoint_bndry':adjoint_bndry,
            'exact_adjoint':exact_adjoint, 'exact_functional': exact_functional, 'calc_functional':calc_functional,
            'choose_output': choose_output}

def advec_diff1D_problem_input (x=None, xl=None, xr=None, n=None):
    def choose_output():
        """Choose which output to see
        Input:  primal: output results of primal problem
                adjoint: output results of adjoint problem
                functional: output results of functional convergence
                all: outputs all three of the above"""
        prob = 'primal'
        # prob = 'adjoint'
        # prob = 'all'
        func_conv = 0
        plot_sol = 1
        plot_err = 0
        show_eig = 1
        return {'prob': prob, 'func_conv': func_conv, 'plot_sol': plot_sol, 'plot_err': plot_err, 'show_eig': show_eig}

    def var_coef_vis (x=1):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        b = 1 # x**0
        return b

    def var_coef_inv (x=1):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        a = 0 # x**0
        return a

    def exact_solution(x):
        u_exact = np.cos(60*x)
        return u_exact

    def boundary_conditions(xl, xr):
        uD_left = np.cos(60 * xl)  # None      # Dirichlet boundary at the left boundary
        uD_right = np.cos(60 * xr)  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
        uN_left = None  # -60*np.sin(60*xl) #None     # Neumann boundary at the left boundary
        uN_right = None  # -60*np.sin(60*xr)  # None    # Neumann boundary at the right boundary
        return {'uD_left': uD_left, 'uD_right': uD_right, 'uN_left': uN_left, 'uN_right': uN_right}

    def source_term(x):
        a = var_coef_inv()
        b = var_coef_vis()
        f = -a*60*np.sin(60*x) + b*3600 * np.cos(60 * x)
        return f

    def adjoint_source_term(x):
        g = np.cos(60*x)
        return g

    def adjoint_bndry (xl, xr):
        psiD_left = 0  # np.cos(60 * xl)  # None      # Dirichlet boundary at the left boundary
        psiD_right = 0  # np.cos(60 * xr)  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
        psiN_left = None  # -60*np.sin(60*xr) #None     # Neumann boundary at the left boundary
        psiN_right = None  # -60*np.sin(60*xl)  # None    # Neumann boundary at the right boundary
        return {'psiD_left': psiD_left, 'psiD_right': psiD_right, 'psiN_left': psiN_left, 'psiN_right': psiN_right}

    def exact_adjoint(x):
        a = var_coef_inv()
        b = var_coef_vis()
        w = 60
        psi = 1/(w*(b**2 * w**2 + a**2)*(np.exp(-a/b)-1)) * ((-b*w*np.cos(w) + a*np.sin(w) + b*w)*np.exp(-a*x/b)\
            + (b*w*np.cos(w*x) - a*np.sin(w*x) - w*b)*np.exp(-a/b) + b*w*np.cos(w) - b*w*np.cos(w*x) - a*np.sin(w) + a*np.sin(w*x))
        return psi

    def exact_functional(xl, xr):
        J_exact = (xr/2 + 1/240 * np.sin(120*xr)) - (xl/2 + 1/240 * np.sin(120*xl))
        return J_exact

    def calc_functional(u, g, h_mat, rx):
        rx_global = np.diag(1 / rx[0, :], 0)  # geometric factor rx = 1/jac
        h_mat_global = sparse.block_diag([h_mat])  # concatenate norm matrix to form global
        rh = sparse.kron(rx_global, h_mat_global)
        J = (np.ones((1, rh.shape[0])) @ rh @ (g * u))[0][0]
        return J

    return {'var_coef_vis': var_coef_vis, 'var_coef_inv': var_coef_inv,'exact_solution': exact_solution, 'boundary_conditions': boundary_conditions,
            'source_term': source_term, 'adjoint_source_term':adjoint_source_term, 'adjoint_bndry':adjoint_bndry,
            'exact_adjoint':exact_adjoint, 'exact_functional': exact_functional, 'calc_functional':calc_functional,
            'choose_output': choose_output}