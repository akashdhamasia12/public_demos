import numpy as np
import sys
n = sys.argv[1]
arr1 = np.random.rand(n, n)
arr2 = np.random.rand(n, n)
arr3 = arr1 * arr2
arr3 = np.log(arr3)
arr3 = np.exp(arr3) 