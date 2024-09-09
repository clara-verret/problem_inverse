import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import norm

class kernel_K:

    def __init__(self, mc_n_): #for now, sphere
        self.mc_n = mc_n_
        self.mc_chord_lenghts = np.array([2*np.sqrt(1-random.random()**2) for _ in range(self.mc_n)]) #random cord length for a sphere of radius 1. multiply by r to get the cord length for a sphere of radius r

    def theoretical(self, l, r):
        return 1-np.sqrt(1-(l/(2*r))**2)
    
    def estimate(self, l, r):
        cord_lenghts = r*self.mc_chord_lenghts
        return np.sum(cord_lenghts < l)/self.mc_n

class CLD:
    def __init__(self, psd_, kernel_k_):
        self.psd = psd_
        self.kernel_k = kernel_k_
        self.kappa=None

    def create_CLD(self,l,r_min, r_max):
        return integrate.quad(lambda r: self.psd(r) * self.kernel_k(l,r), r_min, r_max)[0]
    
    def create_kappa(self, r_min, r_max):
        return self.create_CLD(2*r_max, r_min, r_max)/ integrate.quad(lambda r: self.psd(r), r_min, r_max)[0]
    
    def create_normalized_CLD(self,l, r_min, r_max):
        return self.create_kappa(r_min, r_max) * self.create_CLD(l,r_min, r_max)

###################################################################################################

def solve_direct_problem(num_mc, r_min,r_max,mean,sigma):
    psd = lambda r : norm.pdf(r, mean, sigma)
    list_r=np.linspace(r_min,r_max,100)
    #print(list_r)
    #plt.plot(list_r, [psd(r) for r in list_r])
    #plt.savefig('psd_try.png')
    #plt.show()

    kernel_k =kernel_K(num_mc)
    theory_kernel = lambda l,r : kernel_k.theoretical(l,r)
    estimated_kernel = lambda l,r : kernel_k.estimate(l,r)
    list_l = np.linspace(0,2*list_r[25],100)
    #plt.plot(list_l, [estimated_kernel(l,list_r[25]) for l in list_l], color='r')
    #plt.plot(list_l, [theory_kernel(l,list_r[25]) for l in list_l], color='g')
    #plt.savefig('k_theory_try.png')  
    #plt.show()


    cld = CLD(psd,estimated_kernel)
    estimated_cld = lambda l: cld.create_normalized_CLD(l,r_min, r_max)
    #print(estimated_cld(1.3))
    #plt.plot(list_l, [estimated_cld(l) for l in list_l], color='r')
    #plt.savefig('cld_try.png')  

solve_direct_problem(10000,1,2,1.5,0.3)
exit()