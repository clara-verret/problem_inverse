import random
import numpy as np

#### SPHERES

def below_chord_length_spheres(l,r): #l is the length of the chord, r is the radius of the circle
    """suppose we have a circle of radius r, and we draw* a chord of length l, what is the probability that the chord is below l
    we can use monte carlo to estimate this probability"""
    d = random.uniform(0, r)
    L=2 * np.sqrt(r**2 - d**2)
    return L<l

def k_estimate_shperes(l,r,n): # n is the number of random points
    """monte carlo estimate of k the probability that a chord of length variable L is below l knowing the circle of radius r"""
    count=0
    for _ in range(n):
        if below_chord_length_spheres(l,r):
            count+=1
    return count/n
