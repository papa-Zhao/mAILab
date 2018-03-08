#HW0002
#1. generate five random numbers and print out
#2. generate 10, 100, 1000, 10000, 100000 random numbers and caculate thier means , standard deviations

import numpy as np
import time

five = np.random.random(5)
#five = [1,2,3,4,5]
for i in five:
    print(i)
#print(np.average(five))
#print(np.std(five))
N = 1

for i in range(0,5):
    N = N*10
    #print(N)
    array = np.random.random(N)
    print(str(N)+" average= "+str(np.average(five)))
    print(str(N)+" std= "+str(np.std(five)))