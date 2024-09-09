import random
import numpy as np
import matplotlib.pyplot as plt

#finalement, c'est peut etre plus rapide de faire avec les delta. On peut faire les deux je pense

class K:

    def __init__(self, mc_n_): #for now, sphere
        self.mc_n = mc_n_
        self.mc_chord_lenghts = [2*np.sqrt(1-random.random()**2) for _ in range(self.mc_n)] #random cord length for a sphere of radius 1. multiply by r to get the cord length for a sphere of radius r

    def theoretical(self, l, r):
        return 1-np.sqrt(1-(l/2*r)**2)
    
    def estimate(self, l, r):
        cord_lenghts = r*np.array(self.mc_chord_lenghts)
        return sum(cord_lenghts < l)/self.mc_n

def test_class() :
    array_l=np.linspace(0,50,100)
    k = K(1000)
    y_axis = [k.estimate(l, 18) for l in array_l]

    #plots array_l en x et y_axis en y
    plt.plot(array_l, y_axis)
    plt.show()

test_class()

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