import random
import numpy as np
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
        if l>=2*r:
            return 1
        return 1-np.sqrt(1-(l/(2*r))**2)
    
    def estimate(self, l, r):
        cord_lenghts = r*self.mc_chord_lenghts
        return np.sum(cord_lenghts < l)/self.mc_n

class PSD_to_CLD:
    """
    Takes in a normalized PSD and outputs a CLD

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

    def cumulative_CLD(self,l):
        """Normalized cumulative CLD (Q bar)"""
        return integrate.quad(lambda r: self.psd(r) * self.kernel_k(l,r), self.r_min, self.r_max, epsabs=1e-4, epsrel=1e-4)[0]
    
    def CLD(self,l, h=0.01):
        """Compute CLD using forward finite difference formula for the derivative"""
        return (self.cumulative_CLD(l+h) - self.cumulative_CLD(l-h))/(2*h)

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
    normalized_psd = lambda r : psd(r)/integrate.quad(psd, r_min, r_max)[0]
    #Creates the kernel functions (spheres)
    kernel_k =kernel_K(num_mc)
    #Creates the estimated cumulative CLD
    direct_estimated_k = PSD_to_CLD(normalized_psd,r_min,r_max,kernel_k.estimate)
    #Creates the theoretical cumulative CLD
    direct_theoretical_k = PSD_to_CLD(normalized_psd,r_min,r_max,kernel_k.theoretical)
    return normalized_psd, kernel_k, direct_estimated_k, direct_theoretical_k