import matplotlib.pyplot as plt
import numpy as np
from direct_problem import solve_direct_problem_normal_psd

def analyze_direct_problem(num_mc, r_min,r_max,mean,sigma):
    normalized_psd, kernel_k, direct_estimated_k, direct_theoretical_k= solve_direct_problem_normal_psd(num_mc, r_min,r_max,mean,sigma)
    
    #plot normalized psd
    list_r=np.linspace(r_min,r_max,100)
    plt.plot(list_r, [normalized_psd(r) for r in list_r])
    plt.xlabel('r')
    plt.ylabel('normalized psd')
    plt.savefig('Graphs/normalized_psd.png')
    plt.clf()

    #plot kernel k in fonction of l at fixed r
    r= mean
    list_l = np.linspace(0,2*r,10)
    plt.plot(list_l, [kernel_k.estimate(l,r) for l in list_l], color='r', label='estimated kernel with monte carlo')
    plt.plot(list_l, [kernel_k.theoretical(l,r) for l in list_l], color='g', label='theoretical kernel')
    plt.xlabel('l')
    plt.ylabel('kernel k')
    plt.title('Comparaison of kernels at fixed r')
    plt.legend()
    plt.savefig('Graphs/kernels.png')
    plt.clf()

    #plot cumulative CLD
    list_l = np.linspace(0,2*r_max,10)
    plt.plot(list_l, [direct_estimated_k.cumulative_CLD(l) for l in list_l], color='r', label='estimated cumulative CLD')
    plt.plot(list_l, [direct_theoretical_k.cumulative_CLD(l) for l in list_l], color='g', label='theoretical cumulative CLD')
    plt.xlabel('l')
    plt.ylabel('cumulative CLD')
    plt.title('Comparaison of cumulative CLD')
    plt.legend()
    plt.savefig('Graphs/cumulative_cld.png')
    plt.clf()

    #plot CLD
    plt.plot(list_l, [direct_estimated_k.CLD(l) for l in list_l], color='r', label='estimated CLD')
    plt.plot(list_l, [direct_theoretical_k.CLD(l) for l in list_l], color='g', label='theoretical CLD')
    plt.xlabel('l')
    plt.ylabel('CLD')
    plt.title('Comparaison of CLD')
    plt.legend()
    plt.savefig('Graphs/cld.png')
    plt.clf()
