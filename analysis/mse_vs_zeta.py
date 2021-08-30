import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import wasserstein_distance

def evaluate_MSE_monte_carlo(Nsample,quantizationLevels):
    #data = np.random.triangular(-1,0,1,Nsample)
    data = np.clip(np.random.normal(0,1,Nsample),-6,6)
    quantizedData = np.zeros(Nsample)
    for i in range(Nsample):
        quantizedData[i]=quantizationLevels[np.argmin(np.abs(data[i]-quantizationLevels))]
    #SNR = 10*np.log10(np.true_divide(np.sum(np.var(data)),np.sum(np.var(data-quantizedData))))
    MSE = np.mean(np.square(data-quantizedData))
    return MSE
    
def evaluate_MSE_formula(quantizationLevels,M,pdf,qrange,dx):
    #M = quantizationLevels
    thresholds = 0.5*(np.delete(quantizationLevels,0)+np.delete(quantizationLevels,M-1))
    MSE = 0
    for m in range(M):
        leftThreshold = -6 if m==0 else thresholds[m-1]
        rightThreshold = 6 if m==(M-1) else thresholds[m]
        indices = (qrange>leftThreshold)&(qrange<rightThreshold)
        integrand = np.square((qrange[indices] - quantizationLevels[m]))*pdf[indices]
        MSE += np.sum(integrand*dx)
    return MSE


def placeUniformLevels(M):
    #M is the number of levels
    return np.arange(-1,1.0001,2.0/(M-1))
    #return np.arange(-1,1,2.0/M)

def getNormalLM(M,trials):
    levels = 4*placeUniformLevels(M)
    qrange = np.arange(-6,6,0.0001)
    pdf = norm.pdf(qrange,0,1)
    
    for i in range(trials):
        thresholds = 0.5*(np.delete(levels,0)+np.delete(levels,M-1))
        for m in range(M):
            leftThreshold = -6 if m==0 else thresholds[m-1]
            rightThreshold = 6 if m==(M-1) else thresholds[m]
            indices = (qrange>leftThreshold)&(qrange<rightThreshold)
            numerator = np.sum(pdf[indices]*qrange[indices])
            denominator = np.sum(pdf[indices])
            levels[m] = np.true_divide(numerator,denominator)
    return levels

dx = 0.0001
qrange = np.arange(-6,6,dx)
pdf = norm.pdf(qrange,0,1)
Nsample=1000

zetas = np.arange(1.5,5,0.01).tolist()
M = 64

mses_16 = []
mses_32 = []
mses_64 = []
mses_128 = []
for zeta in zetas:
    #current_levels = zeta*placeUniformLevels(M)
    mses_16.append(evaluate_MSE_formula(zeta*placeUniformLevels(16),16,pdf,qrange,dx))
    mses_32.append(evaluate_MSE_formula(zeta*placeUniformLevels(32),32,pdf,qrange,dx))
    mses_64.append(evaluate_MSE_formula(zeta*placeUniformLevels(64),64,pdf,qrange,dx))
    mses_128.append(evaluate_MSE_formula(zeta*placeUniformLevels(128),128,pdf,qrange,dx))

fig,ax=plt.subplots(figsize=(8,6))

line0, = ax.plot(zetas,mses_16,linewidth=2,label=r'B=4')#color='k',
line1, = ax.plot(zetas,mses_32,linewidth=2,label=r'B=5')#color='k',
line2, = ax.plot(zetas,mses_64,linewidth=2,label=r'B=6')#color='k',
line3, = ax.plot(zetas,mses_128,linewidth=2,label=r'B=7')#color='k',
#line1 = ax.stem(Unif_levels,tick_height*np.ones(M),basefmt=' ',markerfmt=' ',linefmt='r-')
#line2, = ax.plot(BADCS,MPC,linewidth=2,color='g',marker='^',markersize=10,label = 'MPC')
plt.legend(handles=[line0,line1,line2,line3],loc=0,fontsize=20)
ax.grid()
ax.set_xlabel(r'$\zeta$',fontsize=30)
ax.set_ylabel('MSE',fontsize=30)
ax.set_ylim(0.0001,0.03)
ax.set_yscale('log')
#ax.set_title('Lloyd-Max Quantizer',fontsize=30)
#ax.set_xticks(np.arange(1,20))
ax.tick_params(axis='both',labelsize=20)
plt.show()
fig.savefig('mse_v_zeta.svg',bbox_inches='tight')