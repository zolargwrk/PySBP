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
    # u = -(x**3+y**3)
    psi = (x+y)
    # psi = x*(x-1) * y*(y-1)

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

    f = poisson_exact_source_2d(input.u, input.Lxx, input.Lxy, input.Lyx, input.Lyy)
    g = poisson_exact_source_2d(input.psi, input.Lxx, input.Lxy, input.Lyx, input.Lyy)

    uNL = -input.Lxx * (input.u).diff(x) - input.Lxy * (input.u).diff(y)
    uNR = input.Lxx * (input.u).diff(x) + input.Lxy * (input.u).diff(y)
    uNB = -input.Lyx * (input.u).diff(x) - input.Lyy * (input.u).diff(y)
    uNT = input.Lyx * (input.u).diff(x) + input.Lyy * (input.u).diff(y)

    uGradDB = -input.Lyx * (input.u).diff(x) - input.Lyy * (input.u).diff(y)
    uGradDL = -input.Lxx * (input.u).diff(x) - input.Lxy * (input.u).diff(y)
    uGradDT = input.Lyx * (input.u).diff(x) + input.Lyy * (input.u).diff(y)
    uGradDR = input.Lxx * (input.u).diff(x) + input.Lxy * (input.u).diff(y)

    psiNL = -input.Lxx * (input.psi).diff(x) - input.Lxy * (input.psi).diff(y)
    psiNR = input.Lxx * (input.psi).diff(x) + input.Lxy * (input.psi).diff(y)
    psiNB = -input.Lyx * (input.psi).diff(x) - input.Lyy *(input.psi).diff(y)
    psiNT = input.Lyx * (input.psi).diff(x) + input.Lyy * (input.psi).diff(y)

    psiGradDB = -input.Lyx * (input.psi).diff(x) - input.Lyy * (input.psi).diff(y)
    psiGradDL = -input.Lxx * (input.psi).diff(x) - input.Lxy * (input.psi).diff(y)
    psiGradDT = input.Lyx * (input.psi).diff(x) + input.Lyy * (input.psi).diff(y)

    a = 1/8
    b = 1/8

    # the rectangular domain
    bL = 0
    bR = 20
    bB = -5
    bT = 5

    # bL = 0
    # bR = 1
    # bB = 0
    # bT = 1

    # domain type
    domain_type = ['d', 'n', 'd', 'd']
    # domain_type = ['d', 'd', 'd', 'd']

    func_exactVol = sympy.integrate(g*input.u, (x, bL, bR), (y, bB, bT)).subs(m, a).subs(n, b)
    func_exactDB = sympy.integrate(-input.psi.subs(y, bB) * uGradDB.subs(y, bB), (x, bL, bR)).subs(m, a).subs(n, b)
    func_exactDL = sympy.integrate(-input.psi.subs(x, bL) * uGradDL.subs(x, bL), (y, bB, bT)).subs(m, a).subs(n, b)
    func_exactDT = sympy.integrate(-input.psi.subs(y, bT) * uGradDT.subs(y, bT), (x, bL, bR)).subs(m, a).subs(n, b)
    func_exactDR = sympy.integrate(-input.psi.subs(x, bR) * uGradDR.subs(x, bR), (y, bB, bT)).subs(m, a).subs(n, b)
    func_exactNR = sympy.integrate(psiNR.subs(x, bR) * input.u.subs(x, bR), (y, bB, bT)).subs(m, a).subs(n, b)
    func_exactNL = sympy.integrate(psiNL.subs(x, bL) * input.u.subs(x, bL), (y, bB, bT)).subs(m, a).subs(n, b)
    func_exactNB = sympy.integrate(psiNB.subs(y, bB) * input.u.subs(y, bB), (x, bL, bR)).subs(m, a).subs(n, b)
    func_exactNT = sympy.integrate(psiNT.subs(y, bT) * input.u.subs(y, bT), (x, bL, bR)).subs(m, a).subs(n, b)

    if domain_type == ['d', 'n', 'd', 'd']:
        func_exact = (func_exactVol + func_exactDB + func_exactDL + func_exactDT + func_exactNR).evalf()
    elif domain_type == ['d', 'd', 'd', 'd']:
        func_exact = (func_exactVol + func_exactDB + func_exactDL + func_exactDT + func_exactDR).evalf()

    #--------
    # func2_exactVol = sympy.integrate(f * input.psi, (x, 0, 20), (y, -5, 5)).subs(m, a).subs(n, b)
    # func2_exactDB = sympy.integrate(-input.u.subs(y, -5) * psiGradDB.subs(y, -5), (x, 0, 20)).subs(m, a).subs(n, b)
    # func2_exactDL = sympy.integrate(-input.u.subs(x, 0) * psiGradDL.subs(x, 0), (y, -5, 5)).subs(m, a).subs(n, b)
    # func2_exactDT = sympy.integrate(-input.u.subs(y, 5) * psiGradDT.subs(y, 5), (x, 0, 20)).subs(m, a).subs(n, b)
    # func2_exactNR = sympy.integrate(uNR.subs(x, 20) * input.psi.subs(x, 20), (y, -5, 5)).subs(m, a).subs(n, b)
    #
    # func2_exact = (func2_exactVol + func2_exactDB + func2_exactDL + func2_exactDT + func2_exactNR).evalf()
    #----------

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
    # print("func_exact = %.16f" % func2_exact)

    return f, g

poisson_get_exact_2d()


