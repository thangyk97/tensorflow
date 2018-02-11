import numpy as np 
import random
a = np.array([1,2,3,4])
b = np.array([1,2,3,4])
print (a, b)

id_shuffle = np.random.permutation(len(a))
a = a[id_shuffle]
b = b[id_shuffle]

print (a, b)