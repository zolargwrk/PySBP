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
        plot_sol = 0
        plot_err = 0
        show_eig = 0
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
        func_conv = 0
        plot_sol = 0
        plot_err = 0
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
        uD_right = np.cos(30 * xr)  # None     # Dirichlet boundary at the right boundary

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
        psiD_left = -np.cos(30 * xl)/a  # None      # Dirichlet boundary at the left boundary
        psiD_right = np.cos(30 * xr)/a  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary

        return {'psiD_left': psiD_left, 'psiD_right': psiD_right}

    def exact_adjoint(x, xl, xr):
        a = var_coef()
        if a > 0:
            psi = - 1/a * 1/30*np.sin(30*x) + 1/a*1/30*np.sin(30*xr) + 1/a *np.cos(30*xr)
        else:
            psi = - 1/a * 1/30*np.sin(30*x) + 1/a*1/30*np.sin(30*xl) - 1/a *np.cos(30*xl)
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
        # prob = 'primal'
        # prob = 'adjoint'
        prob = 'all'
        func_conv = 1
        plot_sol = 1
        plot_err = 0
        show_eig = 0
        return {'prob': prob, 'func_conv': func_conv, 'plot_sol': plot_sol, 'plot_err': plot_err, 'show_eig': show_eig}

    def var_coef_vis (x=1, b=None):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        if b is None:
            b = 1   # x**0
        return b

    def var_coef_inv (x=1, a=None):
        """variable coefficient evaluated at the nodal location of the scheme of interest.
        Need to get the value of the nodal location x"""
        if a is None:
            a = 0   # x**0
        return a

    def exact_solution(x):
        w = 2*np.pi #30
        u_exact = np.cos(w*x)*np.sin(w*x) #np.cos(w*x)
        return u_exact

    def boundary_conditions(xl, xr):
        w = 2
        uD_left = 0 #np.cos(w * xl)  # None      # Dirichlet boundary at the left boundary
        uD_right = 0 # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
        uN_left = None  # -60*np.sin(60*xl) #None     # Neumann boundary at the left boundary
        uN_right = None #-w*np.sin(w*xr)  # None    # Neumann boundary at the right boundary
        return {'uD_left': uD_left, 'uD_right': uD_right, 'uN_left': uN_left, 'uN_right': uN_right}

    def source_term(x):
        w = 2
        a = var_coef_inv()
        b = var_coef_vis()
        f = 16*np.pi**2* np.cos(2*np.pi*x) * np.sin(2*np.pi*x) #w**2 * b* np.cos(w*x) #-a*60*np.sin(60*x) + b*3600 * np.cos(60 * x)
        return f

    def adjoint_source_term(x):
        b = var_coef_vis()
        g = -20*x**3 #np.cos(30*x) #100*b*np.sin(10*x) #np.cos(60*x)
        return g

    def adjoint_bndry (xl, xr):
        a = var_coef_inv()
        b = var_coef_vis()
        w = 30
        if b!= 0:
            psiD_left = 0 #1/(w**2)*(1 - np.cos(w))  # None      # Dirichlet boundary at the left boundary
            psiD_right = 0 #None  # np.cos(60 * xr)  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
            psiN_left = None  # -60*np.sin(60*xr) #None     # Neumann boundary at the left boundary
            psiN_right = None #1/(w**2)*(1 - w*np.sin(w) - np.cos(w)) #None  # -60*np.sin(60*xl)  # None    # Neumann boundary at the right boundary
        else:
            psiD_left = 0#-np.cos(30 * xl) / a  # None      # Dirichlet boundary at the left boundary
            psiD_right = None#np.cos(30 * xr) / a  # None # np.cos(30*xr)    # Dirichlet boundary at the right boundary
            psiN_left = None  # -60*np.sin(60*xr) #None     # Neumann boundary at the left boundary
            psiN_right = 0#None  # -60*np.sin(60*xl)  # None    # Neumann boundary at the right boundary

        return {'psiD_left': psiD_left, 'psiD_right': psiD_right, 'psiN_left': psiN_left, 'psiN_right': psiN_right}

    def exact_adjoint(x, xl, xr):
        a = var_coef_inv()
        b = var_coef_vis()
        w = 10 #30
        psi = 0
        if a!=0 and b!=0:
            psi = 1/(w*(b**2 * w**2 + a**2)*(np.exp(-a/b)-1)) * ((-b*w*np.cos(w) + a*np.sin(w) + b*w)*np.exp(-a*x/b)\
                + (b*w*np.cos(w*x) - a*np.sin(w*x) - w*b)*np.exp(-a/b) + b*w*np.cos(w) - b*w*np.cos(w*x) - a*np.sin(w) + a*np.sin(w*x))
        elif a==0 and b!=0:
            psi = x**5 - x #1/(w**2)*(np.cos(w*x) + (1 - np.cos(w))*x - np.cos(w)) #-(x**2)/2 + x #1 / (w**2 * b) * np.cos(w * x) + x / (w**2 * b) * (np.cos(w*xl) - np.cos(w*xr)) - np.cos(w*xl) / (w**2 * b)
        elif a!=0 and b==0:
            if a > 0:
                psi = - 1 / a * 1 / w * np.sin(w * x) + 1 / a * 1 / w * np.sin(w * xr) + 1 / a * np.cos(w * xr)
            else:
                psi = - 1 / a * 1 / w * np.sin(w * x) + 1 / a * 1 / w * np.sin(w * xl) - 1 / a * np.cos(w * xl)
        return psi

    def exact_functional(xl, xr):
        w = 30
        b = var_coef_vis()
        # J_exact = b*1/7 * (7*np.cos(50) - 5*np.cos(70) - 2) + 10*b*np.cos(10)*np.cos(60) #(xr/2 + 1/240 * np.sin(120*xr)) - (xl/2 + 1/240 * np.sin(120*xl))
        # J_exact = 8*np.sin(100) + 40*np.sin(20) + 5*np.cos(40) - 5/2 *np.cos(80) - 5/2 + (20*np.cos(20) - 40*np.sin(40))*np.cos(60)
        J_exact = (40*np.pi**2 - 15)/(16*np.pi**3) #1/2 + 1/120 * np.sin(60) + 1/(30**2) * (np.cos(30) - 30*np.sin(30)*np.cos(30) - (np.cos(30))**2) # 1/w * np.sin(w) # # + 1/(30)*(1-np.cos(30))*np.sin(30) #
        return J_exact

    def calc_functional(u, g, h_mat, rx,  db_mat, tr, tl, TD_left, TD_right=None, adj=None):
        b = var_coef_vis()
        rx_global = np.diag(1 / rx[0, :], 0)  # geometric factor rx = 1/jac
        h_mat_global = sparse.block_diag([h_mat])  # concatenate norm matrix to form global
        rh = sparse.kron(rx_global, h_mat_global)
        Dgv = rx[:, 0]*(-(tl.T @ db_mat)) # directional derivative at the left most facet (Dirichlet boundary)
        Dgk = rx[:, -1] * (tr.T @ db_mat)  # directional derivative at the right most facet (Dirichlet boundary)
        nelem = int(len(u)/len(tr))
        nnodes = len(tr)

        if adj==True:
            f = g
            psi = u
            psih = np.reshape(u, (nnodes, nelem), order='F')
            # boundary data
            uD_left = 0 #1
            uN_right = 0 #-30*np.sin(30)
            psiD_left = 0# 1 / (30 ** 2) * (1 - np.cos(30))

            J = (psi.T @ rh @ f)[0][0] \
                - uD_left * (Dgv @ psih[:, 0])[0] \
                + uD_left * TD_left * (tl.T @ psih[:, 0] - psiD_left)[0] \
                + uN_right * (tr.T @ psih[:, -1])[0]
        else:
            uh = np.reshape(u, (nnodes, nelem), order='F')
            # boundary data
            uD_left = 0 #1
            # uD_right = np.cos(30)
            psiD_left = 0 # 1 / (30 ** 2) * (1 - np.cos(30))
            # psiD_right = 1/(30**2) * (1 - np.cos(30))
            psiN = 0 # 1 / (30 ** 2) * (1 - 30 * np.sin(30) - np.cos(30))

            J = (g.T @ rh @ u)[0][0] \
                - psiD_left * (Dgv @ uh[:, 0])[0]\
                + psiD_left * TD_left*(tl.T @ uh[:,0] - uD_left)[0]\
                + psiN * (tr.T @ uh[:, -1])[0]
                # - psiD_right * (Dgk @ uh[:, -1])[0] \
                # + psiD_right * TD_right * (tr.T @ uh[:, -1] - uD_right)[0]

        return J

    return {'var_coef_vis': var_coef_vis, 'var_coef_inv': var_coef_inv,'exact_solution': exact_solution,
            'boundary_conditions': boundary_conditions, 'source_term': source_term,
            'adjoint_source_term':adjoint_source_term, 'adjoint_bndry':adjoint_bndry,
            'exact_adjoint':exact_adjoint, 'exact_functional': exact_functional,
            'calc_functional':calc_functional, 'choose_output': choose_output}