from sympy import *
from sympy import symbols, Matrix, diff
import time
import matplotlib.pyplot as plt
import itertools


x = symbols('x_0 x_1')
fns = {}
fns["himmelblau"] = (x[0]**2 + x[1] + -11)**2 + (x[0] + x[1]**2 - 7)**2
fns["rosenbrock"] = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def SQP(f, nIters, start):
    gradient = Matrix([diff(f,x_i) for x_i in x]) # Symbolic differentiation

    # Initialization
    H = eye(len(x))
    xk = start
    history = []
    gradientX = einsetzen(gradient, xk)

    for k in range(nIters):

        # Calculate next x
        p = - H.inv() * gradientX
        alpha = linesearch(f, p, xk, gradientX)
        xkNext = xk + alpha*p
        if xkNext == xk:
            print("found minimum after", k, "steps")
            return xk, history

        # BFGS for next H approximation
        s = xkNext - xk
        gradientXNext = einsetzen(gradient, xkNext)
        y =  gradientXNext - gradientX
        Hs = H*s
        H = H + (y*y.T) / (y.T*s)[0]  -  Hs*Hs.T / (s.T*Hs)[0]

        # Prepare for next Iteration
        history.append( (xkNext-xk).norm() )
        gradientX = gradientXNext # Just to skip reevaluating
        xk = xkNext.evalf()
        H = H.evalf()

    print(nIters, "steps ecceeded")
    return xk, history

def linesearch(f, p, xk, gradientX):
    c = 0.1
    alpha = 1
    originalValue = einsetzen(f,xk)
    linearFactor = c*alpha*(gradientX.T*p)[0] # [0] to make it scalar
    while true:
        new = einsetzen(f, xk + alpha*p)
        linear = originalValue + alpha * linearFactor
        if new <= linear:
            return alpha
        else:
            alpha /= 2

def einsetzen(fun, vec):
    return fun.subs(list(zip(x, vec)))


def run(fName, start):
    f = fns[fName]
    print("Run", fName, "starting at", start)
    start_time = time.time()
    result, history = SQP(f, 100, start)
    print("x:", result[0], "y:", result[1])
    print("Took", time.time() - start_time, "seconds\n")

    # Plotting
    plt.plot(history)
    plt.ylabel('xkNext - xk')
    plt.yscale("log")
    plt.show()

for fname in fns:
    for start in itertools.product([1.5, -1.5],[4, -4]):
        run(fname, Matrix(start))


# run("himmelblau", Matrix([1.5, 4]))
