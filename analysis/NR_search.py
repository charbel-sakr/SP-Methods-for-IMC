import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
#from scipy.stats import binom
#from scipy.special import comb #n choose k

def NR_method(a0,Niter,B):
    a_n=a0
    for i in range(Niter):
        numerator = np.sqrt(2.0/np.pi)*np.exp(-0.5*np.square(a_n))
        denominator = np.power(2.0,-2.0*B)/3.0+2.0*norm.sf(a_n)
        a_n = numerator/denominator
        print(a_n)
    return a_n

print(NR_method(1.5,20,6))

