import sympy
from types import SimpleNamespace


def poisson_exact_source_2d(u, Lxx, Lxy, Lyx, Lyy):
    x, y = sympy.symbols('x, y')

    dux = u.diff(x)
    duy = u.diff(y)

    f = -(Lxx*dux + Lxy*duy).diff(x) - (Lyx*dux + Lyy*duy).diff(y)

    return f


def poisson_exact_input_2d():
    x, y, m, n = sympy.symbols('x, y, m, n')

    # change u and Lambda to problem definition
    u = sympy.sin(m*sympy.pi*x) * sympy.sin(n*sympy.pi*y)
    # psi = sympy.sin(m*sympy.pi * x) * sympy.cos(n*sympy.pi * y)
    psi = x + y

    Lxx = 4 * x + 1
    Lxy = y
    Lyx = y
    Lyy = y ** 2 + 1
    # Lxx = 1
    # Lxy = 0
    # Lyx = 0
    # Lyy = 1

    return {'u': u, 'psi': psi, 'Lxx': Lxx, 'Lxy': Lxy, 'Lyx': Lyx, 'Lyy': Lyy}


def poisson_get_exact_2d():
    x, y, m, n = sympy.symbols('x, y, m, n')

    input_dict = poisson_exact_input_2d()
    input = SimpleNamespace(**input_dict)

    f = - poisson_exact_source_2d(input.u, input.Lxx, input.Lxy, input.Lyx, input.Lyy)
    g = - poisson_exact_source_2d(input.psi, input.Lxx, input.Lxy, input.Lyx, input.Lyy)

    uNL = -input.Lxx * (input.u).diff(x) - input.Lxy * (input.u).diff(y)
    uNR = input.Lxx * (input.u).diff(x) + input.Lxy * (input.u).diff(y)
    uNB = -input.Lyx * (input.u).diff(x) - input.Lyy * (input.u).diff(y)
    uNT = input.Lyx * (input.u).diff(x) + input.Lyy * (input.u).diff(y)

    uGradDB = -input.Lyx * (input.u).diff(x) - input.Lyy * (input.u).diff(y)
    uGradDL = -input.Lxx * (input.u).diff(x) - input.Lxy * (input.u).diff(y)
    uGradDT = input.Lyx * (input.u).diff(x) + input.Lyy * (input.u).diff(y)

    psiNL = -input.Lxx * (input.psi).diff(x) - input.Lxy * (input.psi).diff(y)
    psiNR = input.Lxx * (input.psi).diff(x) + input.Lxy * (input.psi).diff(y)
    psiNB = -input.Lyx * (input.psi).diff(x) - input.Lyy *(input.psi).diff(y)
    psiNT = input.Lyx * (input.psi).diff(x) + input.Lyy * (input.psi).diff(y)

    func_exactVol = sympy.integrate(g*input.u, (x, 0, 20), (y, -5, 5)).subs(m, 1/3).subs(n, 1/3)

    func_exactDB = sympy.integrate(-input.psi.subs(y, -5) * uGradDB.subs(y, -5), (x, 0, 20)).subs(m, 1/3).subs(n, 1/3)
    func_exactDL = sympy.integrate(-input.psi.subs(x, 0) * uGradDL.subs(x, 0), (y, -5, 5)).subs(m, 1/3).subs(n, 1/3)
    func_exactDT = sympy.integrate(-input.psi.subs(y, 5) * uGradDT.subs(y, 5), (x, 0, 20)).subs(m, 1/3).subs(n, 1/3)
    func_exactNR = sympy.integrate(psiNR.subs(x, 20) * input.u.subs(x, 20), (y, -5, 5)).subs(m, 1/3).subs(n, 1/3)

    func_exact = (func_exactVol + func_exactDB + func_exactDL + func_exactDT + func_exactNR).evalf()

    print("f = ", f)
    print("g = ", g)
    print("uNL = ", uNL)
    print("uNR = ", uNR)
    print("uNB = ", uNB)
    print("uNT = ", uNT)

    print("psiNL = ", psiNL)
    print("psiNR = ", psiNR)
    print("psiNB = ", psiNB)
    print("psiNT = ", psiNT)
    print("func_exact = %.16f" % func_exact)

    return f, g

poisson_get_exact_2d()


