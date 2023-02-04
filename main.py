import numpy as np
import random
a = np.array([[1,1],
              [2,2],
              [3,3]])
np.random.shuffle(a)
for i in a:
    print(i)