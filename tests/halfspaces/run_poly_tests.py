from yroots.polynomial import MultiPower
import yroots as yr
import numpy as np
from time import time
import sys
import utils as ut



def residuals(dim, funcs, roots):
    residuals = []
    for i in range(dim):
        resids_fi = []
        for j in range(len(roots)):
            fi_resids = abs(funcs[i].__call__(roots[j]))[0]
            resids_fi.append(fi_resids)
        residuals.append(resids_fi)
    return residuals



def run_test(dim,deg,num,roots_only = True):
    coeffs = np.load("coeffs/dim{}_deg{}_randn.npy".format(dim,deg))
    a = -np.ones(dim)
    b = np.ones(dim)
    funcs = []
    for coeff in coeffs[num-1]:
        constant_spot = tuple([0]*dim)
        coeff[constant_spot] = 0. # set constant term to zero so all systemss go through origin
        funcs.append(MultiPower(coeff)) # this changes coeffs to a list of functions that we can actually call

    if (roots_only):
        roots = (yr.solve(funcs,a,b))
        return roots
    else:
        start = time()
        roots = (yr.solve(funcs,a,b))
        t = time()-start
        res = residuals(dim,funcs,roots)
        return (res,roots,t)




def run_all_tests(dim, path_name):
    file = "p_tests/dim{}/".format(dim) + path_name + "/"
    rootfile = "p_tests/dim{}/roots/".format(dim)+path_name+"/"
    ut.make_dir("p_tests")
    ut.make_dir("dim{}".format(dim),"p_tests")
    ut.make_dir("roots","p_tests/dim{}".format(dim))
    ut.make_dir(path_name,"p_tests/dim{}".format(dim))
    ut.make_dir(path_name,"p_tests/dim{}/roots".format(dim))
    a = -np.ones(dim)
    b = np.ones(dim)
    all_times = []
    all_res = []
    run_test(dim,2,1)
    highest_degs = [0,0,30,10,7,4,2,2,2,2,2]
    num_tests = [0,0,300,300,300,300,200,200,200,100,100]
    max_res = np.zeros(highest_degs[dim] - 1)
    avg_res = np.zeros(highest_degs[dim] - 1)
    times = np.zeros(highest_degs[dim] - 1)
    for deg in range(2,highest_degs[dim]+1):
        print("running deg {}".format(deg))
        ut.make_dir("deg{}".format(deg),rootfile[:-1])
        all_times = []
        all_res = np.array([])
        num_tests=num_tests[dim]
        for test in range(num_tests):
            #if((test+1)/10 == int((test+1)/10)):
            #    print(f'Dim {dim} Deg {deg}: running test {test+1}/{num_tests}')
            (res,roots,t) = run_test(dim,deg,test+1,False)
            all_times.append(t)
            all_res = np.append(all_res,res)
            roots_file = rootfile + "deg{}/test{}_roots.npy".format(deg,test+1)
            np.save(roots_file,roots)

        all_res = all_res.flatten()
        avg_res[deg-2] = sum(all_res)/len(all_res)
        times[deg-2] = sum(all_times)/len(all_times)
        max_res[deg-2] = max(all_res)
    np.save(file + "avg_resids.txt",avg_res)       #If you ever have to run everyting again, change .txt to .npy lol
    np.save(file +"max_resids.txt",max_res)
    np.save(file + "avg_times.txt",times)
