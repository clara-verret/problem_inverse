import matplotlib.pyplot as plt
import numpy as np

def plot_k(r, k_estimate, ) :
    hist=[]
    list_l=np.linspace(0,2*r,100)

    n=1000
    for l in list_l :
        hist.append(k_estimate(l,r,n))

    plt.plot(list_l,hist,'ro')
    plt.savefig("monte_carlo_estimate.png")
    plt.show()