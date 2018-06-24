from sympy import *
import time
import matplotlib.pyplot as plt
x = Symbol('x')
y = Symbol('y')
fns = {}
fns["himmelblau"] = (x**2 + y + -11)**2 + (x + y**2 - 7)**2
fns["rosenbrock"] = (1 - x)**2 + 100*(y - x**2)**2


def gradientDescent(f, start, gradientF,  nIters, gradientHistory, iIter):
    if nIters <= iIter:
        print(nIters, "steps ecceeded")
        return start, gradientHistory
    gradient = gradientF.subs(start)
    if not any(gradient):
        print("Found minimum after", iIter, "steps")
        return start, gradientHistory
    norm = Matrix(gradient).norm()
    direction = -gradient#/norm
    originalValue = f.subs(start)
    gradientHistory.append(log(originalValue)) # For plotting
#     gradientHistory.append(log(norm)) # to plot gradient instead
    stepsize = 1
#     print(norm, originalValue)
    while (true): # Backtracking linesearch
        nxt = {name: value + stepsize*direction[index] for index, (name, value) in enumerate(start.items())}
        if start == nxt:
            print("No descent step possible", iIter, "steps")
            return start, gradientHistory
        if f.subs(nxt) < originalValue : # Check for descent
            nxt = {n:v.evalf() for n,v in nxt.items()} # Evaluate for performance reasons
            return gradientDescent(f, nxt, gradientF, nIters, gradientHistory, iIter+1)
        else:
            stepsize = stepsize / 2
            if stepsize == 0: raise Exception('Prevented by start==next condition')

def run(fName, start):
    f = fns[fName]
    print("Run", fName, "starting at", start)
    start_time = time.time()
    result, history = gradientDescent(f, start, derive_by_array(f,start), 100, [], 0)
    print(result)
    print("Took", time.time() - start_time)
    # plt.plot(history)
    # plt.ylabel('ln gradient')
    # plt.show()
    print()


run("himmelblau", {x:1.5, y:4})
run("himmelblau", {x:-1.5, y:4})
run("himmelblau", {x:1.5, y:-4})
run("himmelblau", {x:-1.5, y:-4})
run("rosenbrock", {x:1.5, y:4})
run("rosenbrock", {x:-1.5, y:4})
run("rosenbrock", {x:1.5, y:-4})
run("rosenbrock", {x:-1.5, y:-4})
