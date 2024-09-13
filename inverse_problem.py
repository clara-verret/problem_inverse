"""Solve the inverse problem."""

import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.optimize import minimize

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from global_constants import NUM_MC, R_MIN, R_MAX, MEAN, SIGMA
from direct_problem import solve_direct_problem_normal_psd, solve_direct_problem_normal_psd_discrete, PSD_to_CLD

def compute_square(f):
    return lambda x:f(x)**2

def compute_diff(f,g):
    return lambda x :f(x)-g(x)

def loss_tikhonov(observed_cumulative_CLD, kernel_k, gaussian_params,r_min,r_max, delta):
    psd = lambda r : norm.pdf(r, gaussian_params[0], gaussian_params[1])
    normalized_psd = lambda r : psd(r)/integrate.quad(psd, r_min, r_max)[0]
    #print(psd(1))
    direct_CLD = PSD_to_CLD(normalized_psd,r_min,r_max,kernel_k)
    diff_cld = compute_diff(observed_cumulative_CLD,direct_CLD.cumulative_CLD)
    #print(direct_CLD.cumulative_CLD(1))
    #exit()
    return  delta*integrate.quad(psd, 0, 2*r_max)[0] + integrate.quad(compute_square(diff_cld), 0, 2*r_max)[0]

def hyperparameter_tuning(normalized_psd, CLD): 
    list_penalization = np.linspace(0.01, 10, 1000)
    ridge_cv = RidgeCV(alphas=list_penalization, store_cv_values=True)  # store_cv_values=True saves the CV errors for each alpha
    ridge_cv.fit(normalized_psd.reshape(-1, 1), CLD.reshape(-1, 1))
    best_penalization = ridge_cv.alpha_
    print(f"Best alpha (lambda) found: {best_penalization}")
    return best_penalization

####################################################################################################################
def compute_tikhonov(num_mc, r_min,r_max,mean,sigma, mean_initial=1, sigma_initial=0.1, delta=0):
    #create observed cumulative CLD
    normalized_psd, kernel_k, direct_estimated_k, direct_theoretical_k = solve_direct_problem_normal_psd(num_mc, r_min,r_max,mean,sigma)

    #minimize with Tikhonov
    initial_guess= [mean_initial, sigma_initial]
    #print(direct_estimated_k.cumulative_CLD(0))
    result = minimize(lambda gaussian_params: loss_tikhonov(direct_estimated_k.cumulative_CLD, kernel_k.estimate, gaussian_params,r_min,r_max, delta), initial_guess, method = 'Powell')
    print(f'Minimization succed. Mean value : {result.x[0]}. Sigma value : {result.x[1]}')
    return result.x


def compute_discrete_tikhonov(num_mc, r_min,r_max,mean,sigma,noise):
    """Solve the inverse problem using Tikhonov.
    
    Args :
    num_mc : number of monte carlo simulations to estimate the estimated kernel function
    r_min : minimum radius
    r_max : maximum radius
    mean : mean of the PSD
    sigma : standard deviation of the PSD
    noise (bool) : if True add gaussian noise (mean : 0, std : 0.01) to the observed cumulative CLD
    penalization (float) : penalization term in the regression
    """
    normalized_psd, kernel_k, PSD_to_CLD_with_k_estimated, PSD_to_CLD_with_k_theoretical = solve_direct_problem_normal_psd_discrete(num_mc, r_min,r_max,mean,sigma)
    
    #solve Ax=b (see README to understand the formula)
    K = kernel_k.compute_kernel_matrix(r_min,r_max, theortical_computation=True)
    num_l,num_r=np.shape(K)

    Q_with_k_theoretical = PSD_to_CLD_with_k_theoretical.discrete_cumulative_CLD(r_min,r_max)
    Q_with_k_estimated = PSD_to_CLD_with_k_estimated.discrete_cumulative_CLD(r_min,r_max)

    penalization_with_k_theoretical = hyperparameter_tuning(normalized_psd, Q_with_k_theoretical)
    penalization_with_k_estimated = hyperparameter_tuning(normalized_psd, Q_with_k_estimated)
    
    A_with_k_theoretical =  K.T@K + penalization_with_k_theoretical * np.eye(len(K))
    A_with_k_estimated =  K.T@K + penalization_with_k_estimated * np.eye(len(K))

    b_with_k_theoretical= K.T@Q_with_k_theoretical
    b_with_k_estimated= K.T@Q_with_k_estimated

    if noise :
        epsilon = np.random.normal(0, 0.01, num_l)
        b_with_k_theoretical += K.T@epsilon
        b_with_k_estimated += K.T@epsilon

    delta_r=(r_max-r_min)/num_r
    return normalized_psd, np.linalg.solve(A_with_k_theoretical, b_with_k_theoretical)/delta_r, np.linalg.solve(A_with_k_estimated, b_with_k_estimated)/delta_r

