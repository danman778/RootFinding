import numpy as np
import check_roots as cr
import run_2d_tests as d2
import run_3d_tests as d3
import run_4d_tests as d4
import run_poly_tests as rp
import graph_all as gp
import utils as ut

name = "good_roots" # Edit the name of the method we are testing here. VERY IMPORTANT!!!
# If this doesn't change between tests, old data is overwritten

if __name__ == "__main__":
    for dim in range(6,10):
        print("Runnning dim",dim)
        rp.run_all_tests(dim,name)
    
    failed = []
    for dim in range(6,10):
        failed.append(cr.check_p_dim(dim,name))

    ut.make_dir("failed_extra_tests")
    outfile = open("failed_extra_tests/"+name+".txt",'w')
    for item in failed:
        for thing in item:
            outfile.write(str(thing) + "\n")

    gp.make_extra_p_graphs(np.arange(6,10))


    