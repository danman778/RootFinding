import numpy as np
import yroots as yr
import scipy as sp
import matplotlib
from yroots.Combined_Solver import solve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from mpmath import mp, mpmathify
mp.dps = 50

types = ["erik_code","halfspaces","dim_reduction"]
curr = types[0]

file = "t_tests/dim3/" + curr + "/"
rootfile = "t_tests/dim3/roots/"+curr+"/"


def residuals(funcs,roots):
    all_resids = np.array([])
    print(roots)
    for func in funcs:
        all_resids = np.append(all_resids,np.abs(func(roots[:,0],roots[:,1],roots[:,2])))
    return np.mean(all_resids), max(all_resids),roots

def plot_resids(residuals):
    plt.scatter([i+1 for i in range(18)],residuals)
    plt.ylim(1e-20,1e-7)
    plt.xticks(range(1, 19, 2))
    plt.yscale('log')
    plt.axhline(y=2.22044604925031e-13,c='r')
    plt.xlabel('example #')
    plt.ylabel('max residual')
    plt.title('max Residuals for 3d examples (log scale)')
    plt.show()
    return

def newton_polish(funcs, derivs, roots):
    niter = 100
    tol = 1e-32
    new_roots = []

    for root in roots:
        i = 0
        x0, x1 = root, root
        while True:
            if i == niter:
                break
            A = np.array([derivs[j](mp.mpf(x0[0]), mp.mpf(x0[1]), mp.mpf(x0[2]), mp.mpf(x0[3])) for j in range(4)])
            B = np.array([mpmathify(funcs[j](mp.mpf(x0[0]), mp.mpf(x0[1]), mp.mpf(x0[2]), mp.mpf(x0[3]))) for j in range(4)])
            delta = np.array(mp.lu_solve(A, -B))
            norm = mp.norm(delta)
            x1 = delta + x0
            if norm < tol:
                break
            x0 = x1
            i += 1
        new_roots.append(x1)
    return np.array(new_roots)


def ex0():
    f1 = lambda x1,x2,x3 : x1
    f2 = lambda x1,x2,x3 : x1 + x2
    f3 = lambda x1,x2,x3 : x1 + x2 + x3
    funcs = [f1,f2,f3]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start

    return t, residuals(funcs,roots)

def dex0():
    df1 = lambda x1, x2, x3, x4: (1, 0, 0, 0)
    df2 = lambda x1, x2, x3, x4: (1, 1, 0, 0)
    df3 = lambda x1, x2, x3, x4: (1, 1, 1, 0)
    df4 = lambda x1, x2, x3, x4: (1, 1, 1, 1)
    return df1, df2, df3, df4

def ex1(polish=False):
    f1 = lambda x1,x2,x3 : np.sin(x1*x3) + x1*np.log(x2+3) - x1**2
    f2 = lambda x1,x2,x3 : np.cos(4*x1*x2) + np.exp(3*x2/(x1-2)) - 5
    f3 = lambda x1,x2,x3 : np.cos(2*x2) - 3*x3 + 1/(x1-8)
    funcs = [f1,f2,f3]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    if polish:
        roots = polish(funcs, dex1(), roots)
        print(roots)
    return t, residuals(funcs,roots)

def dex1():
    df1 = lambda x1, x2, x3, x4 : (x3*np.cos(x1*x3) + np.log(x2 + 3) - 2*x1, x1/(x2 + 3), x1*np.cos(x1*x3), 0)
    df2 = lambda x1, x2, x3, x4 : (-4*x2*np.sin(4*x1*x2) - (3*x2/(x1 - 2)**2)*np.exp(3*x2/(x1 - 2)), 
                                   -4*x1*np.sin(4*x1*x2) + (3/(x1 - 2))*np.exp(3*x2/(x1 - 2)), 0, 0)
    df3 = lambda x1, x2, x3, x4 : (-1/(x1 - 8)**2, -2*np.sin(2*x2), -3, 0)
    df4 = lambda x1, x2, x3, x4 : (1, 1, -1, -1)
    return df1, df2, df3, df4

def ex2():
    f = lambda x,y,z: np.cosh(4*x*y) + np.exp(z)- 5
    g = lambda x,y,z: x - np.log(1/(y+3))
    h = lambda x,y,z: x**2 -  z
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex2():
    df = lambda x, y, z, x4 : (4*y*np.sinh(4*x*y), 4*x*np.sinh(4*x*y), np.exp(z), 0)
    dg = lambda x, y, z, x4 : (1, 1/(y+3), 0, 0)
    dh = lambda x, y, z, x4 : (2*x, 0, -1, 0)
    df4 = lambda x, y, z, x4 : (1, 1, -1, -1)
    return df, dg, dh, df4

def ex3():
    f = lambda x,y,z: y**2-x**3
    g = lambda x,y,z: (y+.1)**3-(x-.1)**2
    h = lambda x,y,z: x**2 + y**2 + z**2 - 1
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex3():
    df = lambda x, y, z, x4 : (-3*x**2, 2*y, 0, 0)
    dg = lambda x, y, z, x4 : (2*(x-.1), 3*(y+.1)**2, 0, 0)
    dh = lambda x, y, z, x4 : (2*x, 2*y, 2*z, 0)
    df4 = lambda x, y, z, x4 : (1, 1, 1, 1)
    return df, dg, dh, df4

def ex4():
    f = lambda x,y,z: 2*z**11 + 3*z**9 - 5*z**8 + 5*z**3 - 4*z**2 - 1
    g = lambda x,y,z: 2*y + 18*z**10 + 25*z**8 - 45*z**7 - 5*z**6 + 5*z**5 - 5*z**4 + 5*z**3 + 40*z**2 - 31*z - 6
    h = lambda x,y,z: 2*x - 2*z**9 - 5*z**7 + 5*z**6 - 5*z**5 + 5*z**4 - 5*z**3 + 5*z**2 + 1
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex4():
    df = lambda x, y, z, x4 : (0, 0, 22*z**10 + 27*z**8 - 40*z**7 + 15*z**2 - 8*z, 0)
    dg = lambda x, y, z, x4 : (0, 2, 180*z**9 + 200*z**7 - 315*z**6 - 30*z**5 + 25*z**4 - 10*z**3 + 15*z**2 + 80*z - 31, 0)
    dh = lambda x, y, z, x4 : (2, 0, -18*z**8 - 35*z**6 + 30*z**5 - 25*z**4 + 20*z**3 - 15*z**2 + 10*z, 0)
    df4 = lambda x, y, z, x4 : (1, -1, -1, 1)

def ex5():
    f = lambda x,y,z: np.sin(4*(x + z) * np.exp(y))
    g = lambda x,y,z: np.cos(2*(z**3 + y + np.pi/7))
    h = lambda x,y,z: 1/(x+5) - y
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex5():
    df = lambda x, y, z, x4 : (4*np.exp(y)*np.sin(4*(x + z)*np.exp(y)), 4*(x + z)*np.exp(y)*np.cos(4*(x + z)*np.exp(y)), 
                               4*np.exp(y)*np.sin(4*(x + z)*np.exp(y)), 0)
    dg = lambda x, y, z, x4 : (0, -2*np.sin(2*(z**3+y+np.pi/7)), -6*z**2*np.sin(2*(z**3 + y + np.pi/7)), 0)
    dh = lambda x, y, z, x4 : (-1/(x + 5)**2, -1, 0, 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)

#returns known residual of Rosenbrock in 4d
def ex6():
    f = lambda x,y,z : 2*(x-1) - 400*x*(y-x**2)
    g = lambda x,y,z : 2*(y-1) + 200*(y-x**2) - 400*y*(z-y**2)
    h = lambda x,y,z : 200*(z-y**2)
    funcs = [f,g,h]

    a = np.array([-5,-5,-5])
    b = np.array([10,10,10])

    start = time()
    roots = solve(funcs, a, b)
    t = time() - start
    return t, residuals(funcs,roots)

def ex7():
    f = lambda x,y,z: np.cos(10*x*y)
    g = lambda x,y,z: x + y**2
    h = lambda x,y,z: x + y - z
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex7():
    df = lambda x, y, z, x4 : (10*y*np.cos(10*x*y), 10*x*np.cos(10*x*y), 0, 0)
    dg = lambda x, y, z, x4 : (1, 2*y, 0, 0)
    dh = lambda x, y, z, x4 : (1, 1, -1, 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, 1)

def ex8():
    f = lambda x,y,z: np.exp(2*x)-3
    g = lambda x,y,z: -np.exp(x-2*y) + 11
    h = lambda x,y,z: x + y + 3*z
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex8():
    df = lambda x, y, z, x4 : (2*np.exp(2*x), 0, 0, 0)
    dg = lambda x, y, z, x4 : (-np.exp(x - 2*y), 2*np.exp(x - 2*y), 0, 0)
    dh = lambda x, y, z, x4 : (1, 1, 3, 0)
    df4 = lambda x, y, z, x4 : (1, 1, 1, 1)

def ex9():
    f1 = lambda x,y,z: 2*x / (x**2-4) - 2*x
    f2 = lambda x,y,z: 2*y / (y**2+4) - 2*y
    f3 = lambda x,y,z: 2*z / (z**2-4) - 2*z
    funcs = [f1,f2,f3]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex9():
    df1 = lambda x, y, z, x4 : (2*(x**2 + 4)/(x**2 - 4)**2 - 2, 0, 0, 0)
    df2 = lambda x, y, z, x4 : (0, 2*(y**2 - 4)/(y**2 + 4)**2 - 2, 0, 0)
    df3 = lambda x, y, z, x4 : (0, 0, 2*(z**2 + 4)/(z**2 - 4)**2 - 2, 0)
    df4 = lambda x, y, z, x4 : (0, 0, 0, x*(x4**2 + 4)/(x4**2 - 4)**2 - 2)

def ex10():
    f = lambda x,y,z: 2*x**2 / (x**4-4) - 2*x**2 + .5
    g = lambda x,y,z: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
    h = lambda x,y,z: 2*z / (z**2-4) - 2*z
    funcs = [f,g,h]

    start = time()
    roots = solve(funcs, np.array([-1,-1,-1]), np.array([1,1,.8]))
    t = time() - start
    return t, residuals(funcs,roots)

def dex10():
    df = lambda x, y, z, x4 : (-4*x*(x**4 + 4)/(x**4 - 4)**2, 0, 0, 0)
    dg = lambda x, y, z, x4 : (4*x*y/(y**2 + 4) + 2*z, -2*x**2*(y**2 - 4)/(y**2 + 4)**2 - 2, 2*x, 0)
    dh = lambda x, y, z, x4 : (0, 0, -2*(z**2 + 4)/(z**2 - 4)**2 - 2, 0)
    df4 = lambda x, y, z, x4 : (1, 1, 1, 1)
    return df, dg, dh, df4

def ex11():
    f = lambda x,y,z: 144*((x*z)**4+y**4)-225*((x*z)**2+y**2) + 350*(x*z)**2*y**2+81
    g = lambda x,y,z: y-(x*z)**6
    h = lambda x,y,z: (x*z)+y-z
    funcs = [f,g,h]

    start = time()
    roots = solve(funcs,np.array([-1,-1,-2]),np.array([1,1,2]))
    t = time() - start
    return t, residuals(funcs,roots)

def dex11():
    df = lambda x, y, z, x4 : (576*(x**3*z**4) - 450*x*z**2 + 700*x*z**2*y**2, 576*y**3 - 450*y + 700*x**2*z**2*y, 576*x**4*z**3 - 450*x**2*z + 700*x**2*z*y**2, 0)
    dg = lambda x, y, z, x4 : (-6*x**5*z**6, 1, -6*x**6*z**5, 0)
    dh = lambda x, y, z, x4 : (z, 1, x - 1, 0)
    df4 = lambda x, y, z, x4 : (-1, -1, 1, -1)
    return df, dg, dh, df4

def ex12():
    f = lambda x,y,z: x**2+y**2-.49**2
    g = lambda x,y,z: (x-.1)*(x*y - .2)
    h = lambda x,y,z: x**2 + y**2 - z**2
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex12():
    df = lambda x, y, z, x4 : (2*x*y**2, 2*x**2*y, 0, 0)
    dg = lambda x, y, z, x4 : (y*(x - .1) + (x*y-.2), x*(x - .1), 0, 0)
    dh = lambda x, y, z, x4 : (2*x, 2*y, 2*z, 0)
    df4 = lambda x, y, z, x4 : (-1, 1, 1, 1)
    return df, dg, dh, df4

def ex13():
    f = lambda x,y,z: (np.exp(y-z)**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((np.exp(y-z)+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y,z: ((np.exp(y-z)+.4)**3-(x-.4)**2)*((np.exp(y-z)+.3)**3-(x-.3)**2)*((np.exp(y-z)-.5)**3-(x+.6)**2)*((np.exp(y-z)+0.3)**3-(2*x-0.8)**3)
    h = lambda x,y,z: x + y + z
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, a, b)
    t = time() - start
    #roots = np.array([[0,0,0]])
    #t=1000
    #print("0 0 ")
    return t, residuals(funcs,roots)

def ex14():
    f = lambda x,y,z: ((x*z-.3)**2+2*(np.log(y+1.2)+0.3)**2-1)
    g = lambda x,y,z: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    h = lambda x,y,z: x**4 + (np.log(y+1.4)-.3) - z
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex14():
    df = lambda x, y, z, x4 : (2*z*(x*z - .3), 4*(np.log(y + 1.2) + .3)/(y + 1.2), 2*x*(x*z - .3), 0)
    dg = lambda x, y, z, x4 : ((2*(x + .5)*((x - .49)**2 + (y + .5)**2 - 1) + 2*(x - .49)*((x + .5)**2 + (y + .5)**2 - 1))*((x - 1)**2 + (y + .5)**2 - 1) + 2*(x - 1)*(((x - .49)**2 + (y + .5)**2 - 1)*((x + .5)**2 + (y + .5)**2 - 1)),
                               (2*(y + .5)*((x - .49)**2 + (y + .5)**2 - 1) + 2*(y + .5)*((x + .5)**2 + (y + .5)**2 - 1))*((x - 1)**2 + (y + .5)**2 - 1) + 2*(y + .5)*(((x - .49)**2 + (y + .5)**2 - 1)*((x + .5)**2 + (y + .5)**2 - 1)),
                               0, 0)
    dh = lambda x, y, z, x4 : (4*x**3, 1/(y + 1.4), -1, 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)
    return df, dg, dh, df4

def ex15():
    f = lambda x,y,z: np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2))
    g = lambda x,y,z: np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2))
    h = lambda x,y,z: np.exp(x**2*y**2)*np.sin(x-y+z)
    funcs = [f,g,h]

    #f = lambda x,y,z: 0
    #g = lambda x,y,z: 0
    #h = lambda x,y,z: 0

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    #roots = np.array([[0,0,0,0]])
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    #t = 0
    return t, residuals(funcs,roots)

def dex15():
    df = lambda x, y, z, x4 : (10*(1 + y**2)*np.exp(x - 2*x**2 - y**2 - z**2)*np.cos(10*(x + y + z + x*y**2)) + (1 - 4*x)*np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2)),
                            10*(1 + 2*x*y)*np.exp(x - 2*x**2 - y**2 - z**2)*np.cos(10*(x + y + z + x*y**2)) - 2*y*np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2)),
                            10*np.exp(x - 2*x**2 - y**2 - z**2)*np.cos(10*(x + y + z + x*y**2)) - 2*z*np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2)), 0)
    dg = lambda x, y, z, x4 : (10*(1 - 2*y**2)*np.exp(-x+2*y**2+x*y**2*z)*np.cos(10*(x-y-2*x*y**2)) + (-2*x*y**2 + y**2*z)*np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2)),
                            10*(-1-4*x*y)*np.exp(-x+2*y**2+x*y**2*z)*np.cos(10*(x-y-2*x*y**2)) + (-2*x**2*y + 2*x*y*z)*np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2)),
                            x*y**2*np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2)), 0)
    dh = lambda x, y, z, x4 : (-np.exp(x**2*y**2)*np.sin(x-y+z)+2*x*y**2*np.exp(x**2*y**2)*np.cos(x-y+z),
                            np.exp(x**2*y**2)*np.sin(x-y+z)+2*x**2*y*np.exp(x**2*y**2)*np.cos(x-y+z),
                            -np.exp(x**2*y**2)*np.sin(x-y+z), 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)
    return df, dg, dh, df4

def ex16():
    f = lambda x,y,z: ((x-0.1)**2+2*(y*z-0.1)**2-1)*((x*y+0.3)**2+2*(z-0.2)**2-1)
    g = lambda x,y,z: (2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    h = lambda x,y,z: (2*(y+0.1)**2-(z+.15)**2-1)*(2*(x+0.3)**2+(z-.15)**2-1)
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex16():
    df = lambda x, y, z, x4 : (2*y*(x*y+.3)*((x - .1)**2 + 2*(y*z-.1)**2-1) + 2*(x-.01)*((x*y+.3)**2+2*(z-.2)**2-1),
                                2*x*(x*y+.3)*((x - .1)**2 + 2*(y*z-.1)**2-1) + 2*z*(y*z-.1)*((x*y+.3)**2+2*(z-.2)**2-1),
                                2*(z-.2)*((x - .1)**2 + 2*(y*z-.1)**2-1) + 2*y*(y*z-.1)*((x*y+.3)**2+2*(z-.2)**2-1), 0)
    dg = lambda x, y, z, x4 : (2*(x-.21)*(2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1) + 4*z*(x*z+.1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1),
                                4*(y-.15)*(2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1) + ((x-0.21)**2+2*(y-0.15)**2-1)*(2*(y+.1)*(2*(z-0.3)**2+(y-0.15)**2-1)+2*(y-.15)*(2*(x*z+0.1)**2+(y+0.1)**2-1)),
                                4*(z-.3)*(2*(x*z+0.1)**2+(y+0.1)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1) + 4*x*(x*z + .1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1), 0)
    dh = lambda x, y, z, x4 : (4*(x + .3)*(2*(y+0.1)**2-(z+.15)**2-1),
                                4*(y+.1)*(2*(x+0.3)**2+(z-.15)**2-1),
                                2*(z-.15)*(2*(y+0.1)**2-(z+.15)**2-1) + 2*(z+.15)*(2*(x+0.3)**2+(z-.15)**2-1), 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)
    return df, dg, dh, df4

def ex17():
    f = lambda x,y,z: np.sin(3*(x+y+z))
    g = lambda x,y,z: np.sin(3*(x+y-z))
    h = lambda x,y,z: np.sin(3*(x-y-z))
    funcs = [f,g,h]

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve(funcs, -np.ones(3), np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex17():
    df = lambda x, y, z, x4 : (3*np.cos(3*(x+y+z)), 3*np.cos(3*(x+y+z)), 3*np.cos(3*(x+y+z)), 0)
    dg = lambda x, y, z, x4 : (3*np.cos(3*(x+y-z)), 3*np.cos(3*(x+y-z)), -3*np.cos(3*(x+y-z)), 0)
    dh = lambda x, y, z, x4 : (3*np.cos(3*(x-y-z)), -3*np.cos(3*(x-y-z)), -3*np.cos(3*(x-y-z)), 0)
    df4 = lambda x, y, z, x4 : (-1, 1, -1, 1)
    return df, dg, dh, df4

def ex18():
    f = lambda x,y,z: x - 2 + 3*sp.special.erf(z)
    g = lambda x,y,z: np.sin(x*z)
    h = lambda x,y,z: x*y + y**2 - 1
    funcs = [f,g,h]

    a=[-1,-1,-1]
    b=[1,1,1]

    start = time()
    roots = solve(funcs,-np.ones(3),np.ones(3))
    t = time() - start
    return t, residuals(funcs,roots)

def dex18():
    df = lambda x, y, z, x4 : (1, 0, 6/np.sqrt(np.pi)*np.exp(-z**2), 0)
    dg = lambda x, y, z, x4 : (z*np.cos(x*z), 0, x*np.cos(x*z), 0)
    dh = lambda x, y, z, x4 : (y, x + 2*y, 0, 0)
    df4 = lambda x, y, z, x4 : (1, 1, -1, 1)
    return df, dg, dh, df4

if __name__ == "__main__":
    # max_residuals = [ex1(),ex2(),ex3(),ex4(),ex5(),ex6(),ex7(),ex8(),ex9(),ex10(),ex11(),ex12(),ex13(),ex14(),ex15(),ex16(),ex17(),ex18()]
    # plot_resids(max_residuals)
    tests = np.array([ex0,ex1,ex2,ex3,ex4,ex5,ex6,ex7,ex8,ex9,ex10,ex11,ex12,ex13,ex14,ex15,ex16,ex17,ex18])
    times = np.zeros_like(tests)
    avg_resids = np.zeros_like(tests)
    max_resids = np.zeros_like(tests)
    tests[0]()
    for i,test in enumerate(tests):
        t, resids = test()
        avg_resid = resids[0]
        max_resid = resids[1]
        roots = resids[2]
        times[i] = t
        max_resids[i] = max_resid
        avg_resids[i] = avg_resid
        np.save(rootfile+"test_"+str(i)+".npy",roots)
        print("Finished test",i)
    
    np.save(file+"times.npy",times)
    np.save(file+"avg_resids.npy",avg_resids)
    np.save(file+"max_resids.npy",max_resids)