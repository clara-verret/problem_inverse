import random
import numpy as np
import matplotlib.pyplot as plt



#suppose we have a circle of radius r, and we draw a chord of length l, what is the probability that the chord is below l
#we can use monte carlo to estimate this probability
def below_chord_length(l,r): #l is the length of the chord, r is the radius of the circle
    d = random.uniform(0, r)
    L=2 * np.sqrt(r**2 - d**2)
    return L<l

#monte carlo estimate of k the probability that a chord of length variable L is below l knowing the circle of radius r
def k_estimate(l,r,n): # n is the number of random points
    count=0
    for _ in range(n):
        if below_chord_length(l,r):
            count+=1
    return count/n

#run monte carlo
hist=[]
r=1
list_l=np.linspace(0,2*r,100)

n=1000
for l in list_l :
    hist.append(k_estimate(l,r,n))

plt.plot(list_l,hist,'ro')
plt.savefig("monte_carlo_estimate.png")
plt.show()