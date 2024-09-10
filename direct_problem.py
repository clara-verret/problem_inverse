import random
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

class kernel_K:
    """
    gives the theoretical and estimated (with Monte Carlo) kernel function for the direct problem, in the case of spheroids
    kernel function = k(l,r) = P(L<l | R=r)

    Args:
    mc_n_ : number of monte carlo simulations to estimate the estimated kernel function
    eta_ : parameter of the spheroid. eta = 1 for spheres
    """

    def __init__(self, mc_n_, eta_ = 1):
        self.mc_n = mc_n_
        self.mc_chord_lenghts = np.array([2*np.sqrt(1-random.random()**2) for _ in range(self.mc_n)]) #random(see README for precisions) cord length for a sphere of radius 1. multiply by r to get the cord length for a sphere of radius r
        self.eta = eta_

    def theoretical(self, l, r):
        if self.eta == 1: #spheres, we have a closed form solution
            if l >= 2*r:
                return 1
            return 1 - np.sqrt(1-(l/(2*r))**2)

        def alpha(theta, phi) :
            return   ( (np.cos(phi))**2        /       (  (np.cos(theta))**2 + (self.eta**2 * (np.sin(theta))**2 )  ) ) + (np.sin(phi))**2
        
        def integrand(theta, phi):
            sqrtaux = 0 #if the inside of the sqrt is negative, l is bigger than any possible chord, so k(l,r) = 1, so the sqrt is zero
            insidesqrt = 1-(l/(2*r))**2 * alpha(theta, phi)
            if insidesqrt > 0:
                sqrtaux = np.sqrt(insidesqrt)
            return (1-sqrtaux)*np.sin(theta)/(4*np.pi)

        return integrate.dblquad(integrand, 0, 2*np.pi, 0, np.pi)[0]
    
    def estimate(self, l, r):
        '''Monte carlo estimation of the kernel.'''
        cord_lenghts = r*self.mc_chord_lenghts
        return np.sum(cord_lenghts < l)/self.mc_n
    
    def compute_kernel_matrix(self,r_min,r_max, theortical_computation = False, num_l = 100, num_r = 100):
        """Compute the kernel matrix (K(l_i,r_j))_(i,j).
        
        Args : 
        r_min (float) : minimum observed radius
        r_max (float) : maximum observed radius
        theoretical_computation (bool) : True if the kernel k is computed theoretically (Default to True).
        num_l (int) : number of lign of K
        num_r (int) : number of column of K
        """
        kernel_matrix = np.zeros((num_l,num_r))
        radius_vector = np.linspace(r_min,r_max,num_r)
        l_vector = np.linspace(0,2*r_max,num_l)
        if theortical_computation :
            for i,l in enumerate(l_vector) :
                for j,r in enumerate(radius_vector):
                    kernel_matrix[i,j] = self.theoretical(l,r)
        else :
            observed_cord_lengths_vector = np.outer(self.mc_chord_lenghts,radius_vector)
            is_success = observed_cord_lengths_vector[np.newaxis, :, :] < l_vector[:,np.newaxis, np.newaxis]
            kernel_matrix=np.sum(is_success, axis=1)/self.mc_n
        return kernel_matrix

class PSD_to_CLD:
    """
    Takes in a normalized PSD and outputs a CLD

    Args:
    psd_ : function of r
    [r_min_, r_max_] : support of the PSD
    kernel_k_ : function of l and r
    """
    def __init__(self, psd_, r_min_, r_max_, kernel_k_):
        self.psd = psd_
        self.r_min = r_min_
        self.r_max = r_max_
        self.kernel_k = kernel_k_

    def cumulative_CLD(self,l):
        """Normalized cumulative CLD (Q bar)"""
        return integrate.quad(lambda r: self.psd(r) * self.kernel_k(l,r), self.r_min, self.r_max, epsabs=1e-4, epsrel=1e-4)[0]
        
    def CLD(self,l, h=0.01):
        """Compute CLD using forward finite difference formula for the derivative"""
        return (self.cumulative_CLD(l+h) - self.cumulative_CLD(l-h))/(2*h)

class PSD_to_CLD_discrete:
    '''
    From a normalized PSD computes the corresponding CLD in the discrete case.
    
    Args :
    normalized_psd_vector_ (array (Nr,)) : discretization of PSD along radius
    kernel_matrix_ (array (Nl,Nr)) : discretization of the kernel along the chord and the radius
    '''
    def __init__(self, normalized_psd_vector_, kernel_matrix_):
        self.normalized_psd_vector = normalized_psd_vector_
        self.kernel_matrix = kernel_matrix_

    def discrete_cumulative_CLD(self, r_min,r_max):
        _,num_r = np.shape(self.kernel_matrix)
        delta_r=(r_max-r_min)/num_r
        return delta_r * self.kernel_matrix@self.normalized_psd_vector
    

###################################################################################################

def solve_direct_problem_normal_psd(num_mc, r_min,r_max,mean,sigma):
    """
    Solve the direct problem with a normal PSD

    Args:
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the PSD
    sigma : standard deviation of the PSD
    """
    #Creates the kernel functions (spheres)
    kernel_k =kernel_K(num_mc)
    #Creates the PSD and plot it
    psd = lambda r : norm.pdf(r, mean, sigma)
    normalized_psd = lambda r : psd(r)
    #Creates the estimated cumulative CLD
    direct_estimated_k = PSD_to_CLD(normalized_psd,r_min,r_max,kernel_k.estimate)
    #Creates the theoretical cumulative CLD
    direct_theoretical_k = PSD_to_CLD(normalized_psd,r_min,r_max,kernel_k.theoretical)
    return normalized_psd, kernel_k, direct_estimated_k, direct_theoretical_k

def solve_direct_problem_normal_psd_discrete(num_mc, r_min,r_max,mean,sigma):
    """
    Solve the direct problem with a normal PSD

    Args:
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the psd
    sigma : standard deviation of the psd
    """
    #Creates the kernel functions (spheres)
    kernel_k=kernel_K(num_mc)
    #Creates the PSD
    _, num_r = np.shape(kernel_k.compute_kernel_matrix(r_min,r_max))
    psd = lambda r : norm.pdf(r, mean, sigma)
    normalized_discrete_psd = np.array([psd(r) for r in np.linspace(r_min, r_max, num_r)])
    #Creates the discrete estimated cumulative CLD matrix
    PSD_to_CLD_with_k_estimated = PSD_to_CLD_discrete(normalized_discrete_psd,kernel_k.compute_kernel_matrix(r_min,r_max))
    #Creates the discrete theoretical cumulative CLD matrix
    PSD_to_CLD_with_k_theoretical = PSD_to_CLD_discrete(normalized_discrete_psd,
                                               kernel_k.compute_kernel_matrix(r_min,r_max, theortical_computation=True))
    return normalized_discrete_psd, kernel_k, PSD_to_CLD_with_k_estimated, PSD_to_CLD_with_k_theoretical
