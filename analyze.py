import matplotlib.pyplot as plt
from direct_problem import compare_methods

def compare_k():
    array_l, k_estimation, k_theory= compare_methods()
    #plot sur le même graphe