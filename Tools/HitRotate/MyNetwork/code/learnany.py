import numpy as np

a=np.arange(60).reshape(4,5,3)
print(a.shape)

print(a)

b=np.arange(3)

c=a-b
print(c)