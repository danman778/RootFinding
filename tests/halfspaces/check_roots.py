import numpy as np


def sort_roots(orig_roots,idx=0):
    #!!! DON'T USE! sort_roots2() actually works so use that. np.argsort doesn't respect the order stuff goes in. !!!
    #Sorts the roots by first column, then second column, and so on
    num, dim = orig_roots.shape
    #Recursive call: first we sort the last index, then feed the sorted list into the sort of the next index, and so on
    if dim-1 != idx: 
        orig_roots = sort_roots(orig_roots,idx+1)
    #Sort the roots by the current index
    mask = np.argsort(orig_roots[:,idx])
    new_roots = orig_roots[mask]
    return new_roots            

def sort_roots2(orig_roots):
    #Basically jump through hoops to use the np.sort function and just let the order be the indicies
    n = len(orig_roots[0])
    data_types = [("x{}".format(k),float) for k in range(1,n+1)]
    fancy_roots = np.array([tuple(root) for root in orig_roots],data_types)
    new_roots = np.sort(fancy_roots, order = ["x{}".format(k) for k in range(1,n+1)])
    return np.array([np.array(list(root)) for root in new_roots])

def check_t_dim(dim,name,good_name = "good_roots"):
    path = "t_tests/dim{}/roots/".format(dim)
    failed_tests = []
    if dim != 2:
        val = True
        for i in range(19):
            good_roots = np.load(path+good_name+"/test_"+str(i)+".npy")
            curr_roots = np.load(path+name+"/test_"+str(i)+".npy")
            good_roots = np.round(good_roots,12)
            curr_roots = np.round(curr_roots,12)
            good_roots = sort_roots2(good_roots)
            curr_roots = sort_roots2(curr_roots)
            #Compare
            try:
                val = np.allclose(curr_roots,good_roots)
            except:
                val = False
            #print("test",i,"pass:",val)
            if not val:
                failed_tests.append(("t",dim,i))
    else:
        names = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","3.1","3.2","4.1","4.2","5.1","6.1","6.2","6.3","7.1","7.2","7.3","7.4","8.1","8.2","9.1","9.2","10.1"]
        for i,test_name in enumerate(names):
            #Get the roots and sort them
            good_roots = np.load(path+good_name+"/"+test_name+".npy")
            curr_roots = np.load(path+name+"/"+test_name+".npy")
            #Round at 12 decimal places for sort to work right. If the first 12 digits are right, we're probably good for our tests
            curr_roots = np.round(good_roots,12)
            good_roots = np.round(curr_roots,12)
            curr_roots = sort_roots2(good_roots)
            good_roots = sort_roots2(curr_roots)
            #Compare
            try:
                val = np.allclose(curr_roots,good_roots)
            except:
                val = False            #print(name,"pass:",val)
            if not val:
                failed_tests.append(("t",dim,test_name))
    return failed_tests


def check_p_dim(dim,name,good_name = "good_roots"):
    highest_degs = [0,0,30,10,7,4,2,2,2,2,2]
    test_nums = [0,0,300,300,300,300,200,200,200,100,100]
    goodfile = "p_tests/dim{}/roots/".format(dim)+good_name+"/"
    rootfile = "p_tests/dim{}/roots/".format(dim)+name+"/"
    failed_tests = []
    num_tests = test_nums[dim]
    for deg in range(2,highest_degs[dim]+1):
        for test in range(num_tests):
            good_roots = np.load(goodfile+"deg{}/test{}_roots.npy".format(deg,test+1))
            curr_roots = np.load(rootfile+"deg{}/test{}_roots.npy".format(deg,test+1))
            curr_roots = np.round(good_roots,12)
            good_roots = np.round(curr_roots,12)
            curr_roots = sort_roots2(good_roots)
            good_roots = sort_roots2(curr_roots)
            try:
                val = np.allclose(curr_roots,good_roots)
            except:
                val = False
            #print("test",i,"pass:",val)
            if not val:
                failed_tests.append(("p",dim,deg,test+1))
    return failed_tests

