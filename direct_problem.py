import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import norm

class kernel_K:
    """
    gives the theoretical and estimated (with Monte Carlo) kernel function for the direct problem, in the case of a sphere
    kernel function = k(l,r) = P(L<l | R=r)

    Args:
    mc_n_ : number of monte carlo simulations to estimate the estimated kernel function
    """

    def __init__(self, mc_n_):
        self.mc_n = mc_n_
        self.mc_chord_lenghts = np.array([2*np.sqrt(1-random.random()**2) for _ in range(self.mc_n)]) #random cord length for a sphere of radius 1. multiply by r to get the cord length for a sphere of radius r

    def theoretical(self, l, r):
        return 1-np.sqrt(1-(l/(2*r))**2)
    
    def estimate(self, l, r):
        cord_lenghts = r*self.mc_chord_lenghts
        return np.sum(cord_lenghts < l)/self.mc_n

class PSD_to_CLD:
    """
    Takes in a PSD and outputs a CLD

    Args:
    psd_ : function of r
    [r_min_, r_max_] : support of the PSD
    kernel_k_ : function of l and r
    """
    def __init__(self, psd_, rmin_, rmax_, kernel_k_):
        self.psd = psd_
        self.r_min = rmin_
        self.r_max = rmax_
        self.kernel_k = kernel_k_

    def cumulative_CLD(self,l): #This is the normalized cumulative CLD (Q bar)
        return integrate.quad(lambda r: self.psd(r) * self.kernel_k(l,r), self.r_min, self.r_max)[0]
    
    def discrete_CLD(self,l):
        return 0.01*(self.cumulative_CLD(l) - self.cumulative_CLD(l-0.01))

###################################################################################################

def solve_direct_problem_normal_psd(num_mc, r_min,r_max,mean,sigma):
    """
    Solve the direct problem with a normal PSD

    Args:
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the psd
    sigma : standard deviation of the psd
    """
    #Creates the PSD and plot it
    psd = lambda r : norm.pdf(r, mean, sigma)
    list_r=np.linspace(r_min,r_max,100)
    plt.plot(list_r, [psd(r) for r in list_r])
    plt.savefig('Graphs/original_psd.png')
    plt.clf()

    #Creates the kernel functions (spheres) and plots them
    kernel_k =kernel_K(num_mc)
    list_l = np.linspace(0,2*list_r[25],100)
    plt.plot(list_l, [kernel_k.estimate(l,list_r[25]) for l in list_l], color='r')
    plt.plot(list_l, [kernel_k.theoretical(l,list_r[25]) for l in list_l], color='g')
    plt.savefig('Graphs/k_theory_try.png') 
    plt.clf() 
    plt.plot(list_l, [kernel_k.estimate(l,list_r[25]) for l in list_l], color='r')
    plt.savefig('Graphs/k_try.png')
    plt.clf()
    plt.plot(list_l, [kernel_k.theoretical(l,list_r[25]) for l in list_l], color='g')
    plt.savefig('Graphs/k_theory.png')
    plt.clf()

    #Creates the estimated cumulative CLD and plots it
    direct = PSD_to_CLD(psd,r_min,r_max,kernel_k.estimate)
    plt.plot(list_l, [direct.cumulative_CLD(l) for l in list_l], color='r')
    plt.savefig('Graphs/cumulative_cld_try.png')  
    plt.clf()

    #Creates the estimated discrete CLD and plots it
    plt.plot(list_l, [direct.discrete_CLD(l) for l in list_l], color='r')
    plt.savefig('Graphs/discrete_cld_try.png')
    plt.clf()

    #Creates the theoretical cumulative CLD and plots it
    direct = PSD_to_CLD(psd,r_min,r_max,kernel_k.theoretical)
    plt.plot(list_l, [direct.cumulative_CLD(l) for l in list_l], color='g')
    plt.savefig('Graphs/cumulative_cld_theory.png')
    plt.clf()

    #Creates the theoretical discrete CLD and plots it
    plt.plot(list_l, [direct.discrete_CLD(l) for l in list_l], color='g')
    plt.savefig('Graphs/discrete_cld_theory.png')
    plt.clf()

solve_direct_problem_normal_psd(10000,1,2,1.5,0.3)
exit()