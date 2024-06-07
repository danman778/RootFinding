import numpy as np
from time import time
import cProfile
import re

dim_reduce_err = 1e-14
err_lb = 1e-12


def ReduceDimension(A, consts, errs, errors):
    #Get index of first row that has corresponding err less than dim_reduce_err
    A = A.copy()
    row_idx = np.where(errs <= dim_reduce_err)[0][0]
    del_row = A[row_idx]
    col_idx = np.where(abs(del_row.flatten())==np.max(abs(del_row)))[0][0]
    del_norm = del_row[col_idx]
    #largest coeficcient should be larger than the error
    if (abs(del_norm) < err_lb):
        return False,A,consts, errs,errors,None,None,None,None
    temp_consts = consts.copy()
    temp_errs = errs.copy()
    temp_errors = errors.copy()
    temp_consts[row_idx] /= del_norm
    temp_errs[row_idx] /= del_norm
    temp_errors[row_idx] /= del_norm
    del_row /= del_norm
    reduced_A = A-np.matmul(np.vstack(A[:,col_idx]),[del_row])
    new_A = np.vstack([np.hstack([reduced_A[:row_idx,:col_idx],reduced_A[:row_idx,col_idx+1:]]),np.hstack([reduced_A[row_idx+1:,:col_idx],reduced_A[row_idx+1:,col_idx+1:]])])
    reduced_consts = temp_consts - temp_consts[row_idx]*A[:,col_idx]
    new_consts = np.hstack([reduced_consts[:row_idx],reduced_consts[row_idx+1:]])
    reduced_errs = temp_errs + abs(temp_errs[row_idx]*A[:,col_idx])
    new_errs = np.hstack([reduced_errs[:row_idx],reduced_errs[row_idx+1:]])
    reduced_errors = temp_errors + abs(temp_errors[row_idx]*A[:,col_idx])
    new_errors = np.hstack([reduced_errors[:row_idx],reduced_errors[row_idx+1:]])
    return True,new_A,new_consts,new_errs,new_errors,row_idx,col_idx,del_row,del_norm


def ReduceDimFast(A,consts,errs,errors):
    row_idx = np.argmin(errs)
    col_idx = np.argmax(abs(A[row_idx]))
    del_norm = A[row_idx,col_idx]
    if (abs(del_norm) < err_lb):
        return False,A,consts, errs,errors,None,None,None,None
    B = A - A[row_idx,:]*((A[:,col_idx]/del_norm).reshape(-1,1))
    B = np.delete(np.delete(B,row_idx,axis=0),col_idx,axis=1)
    new_consts = np.delete(consts - consts[row_idx]*A[:,col_idx]/del_norm,row_idx)
    new_errs = np.delete(errs + abs(errs[row_idx]*A[:,col_idx]/del_norm),row_idx)
    new_errors = np.delete(errors + abs(errors[row_idx]*A[:,col_idx]/del_norm),row_idx)
    return True,B,new_consts,new_errs,new_errors,row_idx,col_idx,A[row_idx,:],del_norm

def ReduceDimFast2(A,consts,errs,errors):
    row_idx = np.argmin(errs)
    col_idx = np.argmax(abs(A[row_idx]))
    del_norm = A[row_idx,col_idx]
    if (abs(del_norm) < err_lb):
        return False,A,consts, errs,errors,None,None,None,None
    Q = np.hstack([A,consts.reshape(-1,1)])
    R = np.array([errs,errors]).T
    B = Q - Q[row_idx,:]*((Q[:,col_idx]/del_norm).reshape(-1,1))
    B = np.delete(np.delete(B,row_idx,axis=0),col_idx,axis=1)
    C = R + abs(R[row_idx,:]*((A[:,col_idx]/del_norm).reshape(-1,1)))
    C = np.delete(C,row_idx,axis=0)
    return True,B[:,:-1],B[:,-1],C[:,0],C[:,1],row_idx,col_idx,A[row_idx,:]/del_norm,del_norm

def ReduceDimFast3(A,consts,errs,errors):
    row_idx = np.argmin(errs)
    col_idx = np.argmax(abs(A[row_idx]))
    del_norm = A[row_idx,col_idx]
    if (abs(del_norm) < err_lb):
        return False,A,consts, errs,errors,None,None,None,None
    Q = np.hstack([A,consts.reshape(-1,1)])
    R = np.array([errs,errors]).T
    B = Q - Q[row_idx,:]*((Q[:,col_idx]/del_norm).reshape(-1,1))
    B = np.vstack([B[:row_idx,:],B[row_idx+1:,:]])
    B = np.hstack([B[:,:col_idx],B[:,col_idx+1:]])
    C = R + abs(R[row_idx,:]*((A[:,col_idx]/del_norm).reshape(-1,1)))
    C = np.vstack([C[:row_idx,:],C[row_idx+1:,:]])
    return True,B[:,:-1],B[:,-1],C[:,0],C[:,1],row_idx,col_idx,A[row_idx,:]/del_norm,del_norm


def RetrieveDimension(a,b, errs, consts, plane_coeffs, row_idx, col_idx, del_norm):
    bounds = np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    plane_consts = np.hstack([plane_coeffs[:col_idx],plane_coeffs[col_idx+1:]])
    idxs = ((np.sign(plane_consts)+1)/2).astype(int)
    linear_terms = bounds[np.arange(len(idxs)),idxs]
    min = -consts[row_idx]/del_norm - np.dot(linear_terms,plane_consts) - abs(errs[row_idx]/del_norm)
    idxs = 1-idxs
    linear_terms = bounds[np.arange(len(idxs)),idxs]
    max = -consts[row_idx]/del_norm - np.dot(linear_terms,plane_consts) + abs(errs[row_idx]/del_norm)
    minmax = np.array([min,max])
    new_bounds = np.vstack([bounds[:col_idx,:],minmax,bounds[col_idx:,:]])
    return new_bounds[:,0],new_bounds[:,1]

def RetrieveDimFast(a,b,errs,consts,scaled_row,row_idx,col_idx,del_norm):
    bounds = np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    scaled_row = np.delete(scaled_row,col_idx)
    a = np.insert(a,col_idx,sum(np.min(bounds*(-scaled_row.reshape(-1,1)),axis=1)) - consts[row_idx]/del_norm - abs(errs[row_idx]/del_norm))
    b = np.insert(b,col_idx,sum(np.max(bounds*(-scaled_row.reshape(-1,1)),axis=1)) - consts[row_idx]/del_norm + abs(errs[row_idx]/del_norm))
    return a,b

def RetrieveDimFast2(a,b, errs, consts, plane_coeffs, row_idx, col_idx, del_norm):
    bounds = np.hstack([a.reshape(-1,1),b.reshape(-1,1)])
    orig_bounds = bounds.copy()
    plane_consts = np.hstack([plane_coeffs[:col_idx],plane_coeffs[col_idx+1:]])
    mask = np.where(plane_consts > 0)
    bounds[mask] = bounds[mask,::-1]
    new_bounds = -np.sum(bounds*(plane_consts.reshape(-1,1)),axis=0) - consts[row_idx]/del_norm + errs[row_idx]*np.array([-1.,1.])/abs(del_norm)
    final_bounds = np.vstack([orig_bounds[:col_idx,:],new_bounds,orig_bounds[col_idx:,:]])
    return final_bounds[:,0],final_bounds[:,1]

A = np.array([[1.,2.,3.],[2.,3.,2.],[1.,1.,1.]])
consts = np.array([4.,5.,6.])
errs = np.array([1e-12,1e-15,1e-13])
errors = np.array([1e-12,1e-14,1e-8])

shrank,new_A,new_consts,new_errs,new_errs,row_idx,col_idx,scaled_row,del_norm = ReduceDimFast2(A,consts,errs,errors)
a = [-0.8,-1]
b = [1,0.4]

B = np.array([[1.,0.,0.],[1.,1.,0.],[1.,1.,1.]])
consts2 = np.array([1.,1.,1.])
errs2 = np.array([0.,0.2,0.05])
errors2 = np.array([1e-15,1e-12,1e-10])

bounds = np.array([[-.8,1.],[-1.,1.]])
a = bounds[:,0]
b = bounds[:,1]
n = 5
tots = [0.,0.,0.]
funcs = [RetrieveDimension,RetrieveDimFast2, RetrieveDimFast]
for j in range (1):
    #funcs = np.roll(funcs,1)
    #tots = np.roll(tots,1)
    times = []
    times2 = []
    times3 = []
    for i in range(5000):
        A = (np.random.random((n,n)) - 0.5 )* 5
        consts = np.random.random(n)*2 - 1
        errs = np.random.random(n)*1e-7
        errs[n-1] = 0
        errors = np.random.random(n)*1e-13
        a = np.random.random(n-1) - 1.
        b = np.random.random(n-1)
        shrank,new_A,new_consts,new_errs,new_errs,row_idx,col_idx,scaled_row,del_norm  = ReduceDimFast2(A,consts,errs,errors)

        t = time()
        a2,b2 = funcs[2](a,b,errs,consts,scaled_row,row_idx,col_idx,del_norm)
        end2 = time() - t
        if i > 3000:
            times2.append(end2)

        t = time()
        a3,b3 = funcs[1](a,b,errs,consts,scaled_row,row_idx,col_idx,del_norm)
        end3 = time() - t
        if i > 3000:
            times3.append(end3)

        t = time()
        a1,b1 = funcs[0](a,b,errs,consts,scaled_row,row_idx,col_idx,del_norm)
        end = time() - t
        if i > 3000:
            times.append(end)

        assert np.allclose(a1,a2)
        assert np.allclose(b1,b2)
        assert np.allclose(a1,a3)
        assert np.allclose(b1,b3)

    tots[0] += sum(times)
    tots[1] += sum(times3)
    tots[2] += sum(times2)
print(tots)


times = []
times2 = []
times3 = []
times4 = []
n = 2
for i in range(1000):
    A = (np.random.random((n,n)) - 0.5 )* 5
    consts = np.random.random(n)
    errs = np.random.random(n)*1e-7
    errs[n-1] = 0
    errors = np.random.random(n)*1e-13

    t = time()
    res1 = ReduceDimFast(A,consts,errs,errors)
    end = time() - t
    times2.append(end)

    t = time()
    res4 = ReduceDimFast3(A,consts,errs,errors)
    end4 = time() - t
    times4.append(end4)

    t = time()
    res2 = ReduceDimFast2(A,consts,errs,errors)
    end = time() - t
    times3.append(end)

    t = time()
    res3 = ReduceDimension(A,consts,errs,errors)
    end2 = time() - t
    times.append(end2)

    if not (np.allclose(res1[1],res2[1])) or not(np.allclose(res2[2],res3[2])) or not (np.allclose(res4[1],res3[1])):
        print("bad")

print(sum(times),sum(times2),sum(times3),sum(times4))