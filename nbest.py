from numpy.random import randint as rd
import numpy as np
np.random.seed(0)

l = np.array([rd(0,100) for i in range(20)])
print(l.argsort()[:10])
print(l)