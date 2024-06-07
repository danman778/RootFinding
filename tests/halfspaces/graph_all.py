import numpy as np
from matplotlib import pyplot as plt
import os

def stacktoarray(list):
    list = list.replace(" ","")
    list = list.replace("*^","e")
    list = list.split("\n")
    while (list[-1] == ''):
        list.pop()
    for i in range(len(list)):
        list[i] = abs(float(list[i]))
    return list

def getArgName(name):
    if name == "times":
        return "Times"
    if name == "avg_times":
        return "Average Times"
    if name == "avg_resids":
        return "Average Residuals"
    if name == "max_resids":
        return "Maximum Residuals"


def make_p_graphs():
    
    max_degs = [0,0,30,10,7,4]
    args = ["avg_times","avg_resids","max_resids"]
    dims = np.arange(2,6)

    for dim in dims:
        filesList = np.array(os.listdir("p_tests/dim{}".format(dim)))
        types = filesList[np.where([os.path.isdir("p_tests/dim{}/{}".format(dim,dir)) for dir in filesList])]
        print(types)
        for arg in args:
            fig,ax1 = plt.subplots()
            ax1.set_yscale("log")
            data = []
            for name in types:
                if name == "roots":
                    continue
                curr_data = np.load("p_tests/dim{}/{}/{}.txt.npy".format(dim,name,arg))
                data.append(curr_data)
                X = np.arange(2,min(len(curr_data),max_degs[dim])+2)
                ax1.plot(X,curr_data,label = name)
            if (arg == "avg_times"):
                ax1.set_ylabel("Time Log Scale")
            else:
                ax1.set_ylabel("Residual")
            plt.title("Dim {} {}".format(dim,getArgName(arg)))
            ax1.set_xlabel("Degree")
            plt.tight_layout()
            ax1.legend()
            plt.savefig("graphs/p_tests/dim{}_{}.png".format(dim,arg))
            plt.clf()
            plt.close()


def make_extra_p_graphs():
    args = ["avg_times","avg_resids","max_resids"]
    dims = np.arange(6,10)

    for arg in args:
        fig,ax1 = plt.subplots()
        ax1.set_yscale("log")
        filesList = np.array(os.listdir("p_tests/dim{}".format(dims[0])))
        types = filesList[np.where([os.path.isdir("p_tests/dim{}/{}".format(dim,dir)) for dir in filesList])]
        for name in types:
            if name == "roots":
                continue
            data = []
            for dim in dims:
                curr_data = np.load("p_tests/dim{}/{}/{}.txt.npy".format(dim,name,arg))
                data.append(curr_data[0])
            ax1.plot(dims,data,label = name)
        if (arg == "avg_times"):
            ax1.set_ylabel("Time Log Scale")
        else:
            ax1.set_ylabel("Residual")
        plt.title("High Dim {} (Degree 2 Only)".format(getArgName(arg)))
        ax1.set_xlabel("Dim")
        plt.tight_layout()
        ax1.legend()
        plt.savefig("graphs/p_tests/high_dim_{}.png".format(dim,arg))
        plt.clf()
        plt.close()

def make_t_graphs():
    dim_2_names = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","3.1","3.2","4.1","4.2","5.1","6.1","6.2","6.3","7.1","7.2","7.3","7.4","8.1","8.2","9.1","9.2","10.1"]
    args = ["times","avg_resids","max_resids"]
    dims = np.arange(2,5)

    for dim in dims:
        filesList = np.array(os.listdir("t_tests/dim{}".format(dim)))
        types = filesList[np.where([os.path.isdir("t_tests/dim{}/{}".format(dim,dir)) for dir in filesList])]
        if dim == 2:
            names = dim_2_names
        else:
            names = [str(i) for i in range(19)]
        for arg in args:
            fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
            ax.set_yscale("log")
            data = []
            width = 1/len(types)  
            multiplier = 0
            for name in types:
                if name == "roots":
                    continue
                curr_data = np.load("t_tests/dim{}/{}/{}.npy".format(dim,name,arg),allow_pickle = True)
                data.append(curr_data)
                #X = np.arange(2,min(len(curr_data),max_degs[dim])+2)
                #ax1.plot(X,curr_data,label = name)    
                x = np.arange(len(curr_data)) 
                offset = width * multiplier
                ax.bar(x + offset, curr_data, width, label=name)
                multiplier += 1
            
            ax.set_xticks(x + 1.5*width, names,rotation=45)
            if (arg == "times"):
                ax.set_ylabel("Time Log Scale")
            else:
                ax.set_ylabel("Residual")
                ax.set_ylim([1e-17,1e-6])
            plt.title("Dim {} {}".format(dim,getArgName(arg)))
            ax.set_xlabel("Test")
            plt.tight_layout()
            ax.legend()
            plt.savefig("graphs/t_tests/dim{}_{}.png".format(dim,arg))
            plt.clf()


if __name__ == "__main__":
    make_p_graphs()
    make_t_graphs()
        

