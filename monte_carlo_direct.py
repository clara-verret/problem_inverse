import random
import numpy as np

#### SPHERES

def below_chord_length_spheres(l,r): #l is the length of the chord, r is the radius of the circle
    """
    r = radius of the circle
    l = length of the chord

    Generates a random(see README for details) chord and returns if the chord is of length less than l.
    """
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
