import random
import numpy as np

N=100000
discretisation_step = 100

def k_theoretical(array_l,r, eta):
    return 1-np.sqrt(1-(array_l/2*r)**2)

#### SPHERES

###Global variables
R=1

def below_chord_length_spheres(array_l,r): #l is the length of the chord, r is the radius of the circle
    """Generates a random(see README for details) chord and returns if the chord is of length less than l.

    Args :
    r = radius of the circle
    l = length of the chord
    """
    d = random.uniform(0, r)
    L=2 * np.sqrt(r**2 - d**2)
    return L<array_l

def k_estimate_spheres_mc(array_l,r,n): # n is the number of random points
    """For multiples length l, estimates the probability of a chord being less than l in length (kernel k), by generating n random chords and using Monte Carlo method.

    Args :
    r = radius of the circle
    array_l (1d-array) : array of length of the chord
    n (int) : number of generated chords

    Return:
    Return 1d-array of estimated probabilities.
"""
    count=np.zeros(len(array_l))
    for _ in range(n):
        is_successful = below_chord_length_spheres(array_l,r)
        for index, value in enumerate(is_successful):
            if value :
                count[index]+=1
    return count/n

def compare_methods():
    """Compare theoretical and estimated k.
    """
    array_l=np.linspace(0,2*R,discretisation_step)
    k_estimation=k_estimate_spheres_mc(array_l,R,N)
    k_theory = k_theoretical(array_l,R, 1)
    return array_l,k_estimation, k_theory