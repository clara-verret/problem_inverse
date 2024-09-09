import matplotlib.pyplot as plt
from monte_carlo_direct import compare_methods

def compare_k():
    array_l, k_estimation, k_theory= compare_methods()
    #plot sur le même graphe