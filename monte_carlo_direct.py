import random
import numpy as np
import matplotlib.pyplot as plt

N=100000

#### SPHERES

###Global variables
R=1

def below_chord_length_spheres(array_l,r): #l is the length of the chord, r is the radius of the circle
    """suppose we have a circle of radius r, and we draw* a chord of length l, what is the probability that the chord is below l
    we can use monte carlo to estimate this probability"""
    d = random.uniform(0, r)
    L=2 * np.sqrt(r**2 - d**2)
    return L<array_l

def k_estimate_spheres(array_l,r,n): # n is the number of random points
    """monte carlo estimate of k the probability that a chord of length variable L is below l knowing the circle of radius r"""
    count=np.zeros(len(array_l))
    for _ in range(n):
        is_successful = below_chord_length_spheres(array_l,r)
        for index, value in enumerate(is_successful):
            if value :
                count[index]+=1
    return count/n

def main_monte_carlo():
    """main function to compute the monte carlo estimate of k"""
    list_l=np.linspace(0,2*R,100)
    k_estimation=k_estimate_spheres(list_l,R,N)
    #plot the result
    plt.plot(list_l,k_estimation,'ro')
    plt.xlabel("l")
    plt.ylabel("k for r=1")
    plt.savefig("monte_carlo_estimate_k_sphere.png")

    return list_l,k_estimation