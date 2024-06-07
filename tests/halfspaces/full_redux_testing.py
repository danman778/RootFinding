import numpy as np
import yroots as yr

f = lambda x1,x2,x3,x4: x1
g = lambda x1,x2,x3,x4: x2
h = lambda x1,x2,x3,x4: np.cos(x1+x2)+x3**3
f4 = lambda x1,x2,x3,x4: x4**(3)

roots = yr.solve([g,f,h,f4],-np.ones(4),np.ones(4))
print(roots)
print(g(*roots[0]))
print(f(*roots[0]))
print(h(*roots[0]))
print(f4(*roots[0]))