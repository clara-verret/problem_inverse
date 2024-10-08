'''Plot and analyze the outputs of direct problem and inverse problem'''

import matplotlib.pyplot as plt
import numpy as np

from direct_problem import solve_direct_problem_normal_psd, solve_direct_problem_normal_psd_discrete, solve_direct_problem_dirac_psd
from inverse_problem import compute_discrete_tikhonov

def analyze_direct_problem(num_mc, r_min,r_max,mean,sigma):
    """Plot the functions PSD, kernel K, cumulative CLD and CLD.
    
    Args :
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the PSD
    sigma : standard deviation of the PSD
    """
    normalized_psd, kernel_k, direct_estimated_k, direct_theoretical_k= solve_direct_problem_normal_psd(num_mc, r_min,r_max,mean,sigma)
    
    #plot normalized PSD
    list_r=np.linspace(r_min,r_max,100)
    plt.plot(list_r, [normalized_psd(r) for r in list_r])
    plt.xlabel('r')
    plt.ylabel('normalized PSD')
    plt.savefig('Graphs/normalized_psd.png')
    plt.clf()

    #plot kernel k in fonction of l at fixed r
    r= mean
    list_l = np.linspace(0,4*r,10)
    plt.plot(list_l, [kernel_k.estimate(l,r) for l in list_l], color='r', label='estimated kernel with monte carlo')
    plt.plot(list_l, [kernel_k.theoretical(l,r) for l in list_l], color='g', label='theoretical kernel')
    plt.xlabel('l')
    plt.ylabel('kernel k')
    plt.title('Comparison of kernels at fixed r')
    plt.legend()
    plt.savefig('Graphs/kernels.png')
    plt.clf()

    #plot cumulative CLD
    list_l = np.linspace(0,2*r_max,10)
    plt.plot(list_l, [direct_estimated_k.cumulative_CLD(l) for l in list_l], color='r', label='estimated cumulative CLD')
    plt.plot(list_l, [direct_theoretical_k.cumulative_CLD(l) for l in list_l], color='g', label='theoretical cumulative CLD')
    plt.xlabel('l')
    plt.ylabel('cumulative CLD')
    plt.title('Comparison of cumulative CLD')
    plt.legend()
    plt.savefig('Graphs/cumulative_cld.png')
    plt.clf()

    #plot CLD
    plt.plot(list_l, [direct_estimated_k.CLD(l) for l in list_l], color='r', label='estimated CLD')
    plt.plot(list_l, [direct_theoretical_k.CLD(l) for l in list_l], color='g', label='theoretical CLD')
    plt.xlabel('l')
    plt.ylabel('CLD')
    plt.title('Comparison of CLD')
    plt.legend()
    plt.savefig('Graphs/cld.png')
    plt.clf()

def analyze_direct_problem_dirac(num_mc, r):
    theory_CLD, kernel_k_theory = solve_direct_problem_dirac_psd(num_mc, r, mode = 'theory')
    estimated_CLD, kernel_k_estimated = solve_direct_problem_dirac_psd(num_mc, r, mode = 'estimated')

    #plot kernel k
    list_l = np.linspace(0,4*r,10)
    plt.plot(list_l, [kernel_k_estimated.function(l,r) for l in list_l], color='r', label='estimated kernel with monte carlo')
    plt.plot(list_l, [kernel_k_theory.function(l,r) for l in list_l], color='g', label='theoretical kernel (rectangle method)')
    plt.xlabel('l')
    plt.ylabel('kernel k')
    plt.title('Comparison of kernels at fixed radius')
    plt.legend()
    plt.savefig('Graphs/kernels_dirac.png')
    plt.clf()

    #plot cumulative CLD
    list_l = np.linspace(0,2*r,10)
    plt.plot(list_l, [estimated_CLD.cumulative_CLD(l) for l in list_l], color='r', label='estimated cumulative CLD')
    plt.plot(list_l, [theory_CLD.cumulative_CLD(l) for l in list_l], color='g', label='theoretical cumulative CLD')
    plt.xlabel('l')
    plt.ylabel('cumulative CLD')
    plt.title('Comparison of cumulative CLD')
    plt.legend()
    plt.savefig('Graphs/cumulative_cld_dirac.png')
    plt.clf()

    #plot CLD
    plt.plot(list_l, [estimated_CLD.CLD(l) for l in list_l], color='r', label='estimated CLD')
    plt.plot(list_l, [theory_CLD.CLD(l) for l in list_l], color='g', label='theoretical CLD')
    plt.xlabel('l')
    plt.ylabel('CLD')
    plt.title('Comparison of CLD')
    plt.legend()
    plt.savefig('Graphs/cld_dirac.png')
    plt.clf()

def analyze_direct_problem_dirac_eta(num_mc, r):
    etas = [0.5,1,2]

    theory_cld_eta_05, kernel_k_theory_eta_05 = solve_direct_problem_dirac_psd(num_mc, r, mode = 'theory', eta = 0.5)
    theory_cld_eta_1, kernel_k_theory_eta_1 = solve_direct_problem_dirac_psd(num_mc, r, mode = 'theory', eta = 1)
    theory_cld_eta_2, kernel_k_theory_eta_2 = solve_direct_problem_dirac_psd(num_mc, r, mode = 'theory', eta = 2)

    #plot kernel k
    list_l = np.linspace(0,4*r,10)
    plt.plot(list_l, [kernel_k_theory_eta_05.function(l,r) for l in list_l], color='r', label='eta = 0.5')
    plt.plot(list_l, [kernel_k_theory_eta_1.function(l,r) for l in list_l], color='g', label='eta = 1')
    plt.plot(list_l, [kernel_k_theory_eta_2.function(l,r) for l in list_l], color='b', label='eta = 2')
    plt.xlabel('l')
    plt.ylabel('kernel k')
    plt.title('Comparison of kernels at fixed r')
    plt.legend()
    plt.savefig('Graphs/kernels_dirac_eta.png')
    plt.clf()

    #plot cumulative CLD
    list_l = np.linspace(0,4*r,10)
    plt.plot(list_l, [theory_cld_eta_05.cumulative_CLD(l) for l in list_l], color='r', label='eta = 0.5')
    plt.plot(list_l, [theory_cld_eta_1.cumulative_CLD(l) for l in list_l], color='g', label='eta = 1')
    plt.plot(list_l, [theory_cld_eta_2.cumulative_CLD(l) for l in list_l], color='b', label='eta = 2')
    plt.xlabel('l')
    plt.ylabel('cumulative CLD')
    plt.title('Comparison of cumulative CLD')
    plt.legend()
    plt.savefig('Graphs/cumulative_cld_dirac_eta.png')
    plt.clf()

    #plot CLD
    plt.plot(list_l, [theory_cld_eta_05.CLD(l) for l in list_l], color='r', label='eta = 0.5')
    plt.plot(list_l, [theory_cld_eta_1.CLD(l) for l in list_l], color='g', label='eta = 1')
    plt.plot(list_l, [theory_cld_eta_2.CLD(l) for l in list_l], color='b', label='eta = 2')
    plt.xlabel('l')
    plt.ylabel('CLD')
    plt.title('Comparison of CLD')
    plt.legend()
    plt.savefig('Graphs/cld_dirac_eta.png')
    plt.clf()

def analyze_discrete_direct_problem(num_mc, r_min,r_max,mean,sigma):
    """Plot the matrix PSD, kernel K, cumulative CLD.
    
    Args :
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the PSD
    sigma : standard deviation of the PSD
    """
    normalized_psd, kernel_k, direct_estimated_k, direct_theoretical_k= solve_direct_problem_normal_psd_discrete(num_mc, r_min,r_max,mean,sigma)
    num_l, num_r = np.shape(kernel_k.compute_kernel_matrix(r_min,r_max))
    
    #plot kernel k in fonction of l at fixed r
    l_vector = np.linspace(0,2*r_max,num_l)
    plt.plot(l_vector,kernel_k.compute_kernel_matrix(r_min,r_max)[:,-1], color='r', label='estimated kernel with monte carlo')
    plt.plot(l_vector,kernel_k.compute_kernel_matrix(r_min,r_max, theortical_computation=True)[:,-1], color='g', label='theoretical kernel')
    plt.xlabel('l')
    plt.ylabel('kernel k')
    plt.title(f'Comparison of kernels at fixed r')
    plt.legend()
    plt.savefig('Graphs/discrete_kernels.png')
    plt.clf()

    #plot normalized PSD
    list_r=np.linspace(r_min,r_max,num_r)
    plt.plot(list_r, normalized_psd)
    plt.xlabel('r')
    plt.ylabel('normalized PSD')
    plt.savefig('Graphs/normalized_discrete_psd.png')
    plt.clf()

    #plot cumulative CLD
    list_l = np.linspace(0,2*r_max,num_l)
    plt.plot(list_l, direct_estimated_k.discrete_cumulative_CLD(r_min,r_max), color='r', label='estimated cumulative CLD')
    plt.plot(list_l, direct_theoretical_k.discrete_cumulative_CLD(r_min,r_max), color='g', label='theoretical cumulative CLD')
    plt.xlabel('l')
    plt.ylabel('cumulative CLD')
    plt.title('Comparison of cumulative CLD')
    plt.legend()
    plt.savefig('Graphs/cumulative_cld.png')
    plt.clf()

def analyze_discrete_inverse_problem(num_mc, r_min,r_max,mean,sigma, noise):
    """Plot the matrix PSD, infered PSD when CLD is computed with theoretical formula of k,
    infered PSD when CLD is computed with monte carlo estimation of k.
    
    Args :
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the PSD
    sigma : standard deviation of the PSD
    noise (bool) : if True add noise to the observed cumulative CLD
    """
    true_psd, estimated_psd_with_k_theoretical, estimated_psd_with_k_estimated  = compute_discrete_tikhonov(num_mc, r_min,r_max,mean,sigma, noise)
    
    #plot infered PSD and compare it with the true PSD
    radius_vector = np.linspace(r_min,r_max,len(true_psd))
    plt.plot(radius_vector,true_psd, color='g', label='true PSD')
    plt.plot(radius_vector,estimated_psd_with_k_theoretical, color='r', label='infered PSD with k_theo')
    plt.plot(radius_vector,estimated_psd_with_k_estimated, color='b', label='infered PSD with k_MC')
    plt.xlabel('r')
    plt.ylabel('PSD')
    plt.title('Infered PSD with Tikhonov method')
    plt.legend()
    plt.savefig('Graphs/infered_psd.png')
    plt.clf()

    #compute L2 norm
    print('for psd with theoretical computation of k : || psi_infered - psi_tru||', np.linalg.norm(true_psd - estimated_psd_with_k_theoretical))
    print('for psd with monte carlo computation of k : || psi_infered - psi_tru||', np.linalg.norm(true_psd - estimated_psd_with_k_estimated))