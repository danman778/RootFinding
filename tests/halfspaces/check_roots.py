import numpy as np

dim = 3

names = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","3.1","3.2","4.1","4.2","5.1","6.1","6.2","6.3","7.1","7.2","7.3","7.4","8.1","8.2","9.1","9.2","10.1"]
types = ["erik_code/","halfspaces/","dim_reduction/"]
path = "t_tests/dim{}/roots/".format(dim)


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


if dim == 3:
    for i in range(19):
        e_roots = np.load(path+types[0]+"test_"+str(i)+".npy")
        h_roots = np.load(path+types[1]+"test_"+str(i)+".npy")
        d_roots = np.load(path+types[2]+"test_"+str(i)+".npy")
        e_roots = np.round(e_roots,12)
        d_roots = np.round(d_roots,12)
        h_roots = np.round(h_roots,12)
        e_roots = sort_roots2(e_roots)
        d_roots = sort_roots2(d_roots)
        h_roots = sort_roots2(h_roots)
        #Compare
        val = np.allclose(d_roots,e_roots) and np.allclose(e_roots,h_roots)
        print("test",i,"pass:",val)
        if not val:
            print(e_roots)
            print(d_roots)
            print(h_roots)
            print(e_roots-d_roots)
else:
    for name in names:
        #Get the roots and sort them
        e_roots = np.load(path+types[0]+name+".npy")
        h_roots = np.load(path+types[1]+name+".npy")
        d_roots = np.load(path+types[2]+name+".npy")
        #Round at 12 decimal places for sort to work right. If the first 12 digits are right, we're probably good for our tests
        e_roots = np.round(e_roots,12)
        d_roots = np.round(d_roots,12)
        h_roots = np.round(h_roots,12)
        e_roots = sort_roots(e_roots)
        d_roots = sort_roots(d_roots)
        h_roots = sort_roots(h_roots)
        #Compare
        val = np.allclose(d_roots,e_roots) and np.allclose(e_roots,h_roots)
        print(name,"pass:",val)
        if not val:
            print(e_roots)
            print(d_roots)
            print(h_roots)
