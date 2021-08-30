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
    
def evaluate_MSE_formula(quantizationLevels,pdf,qrange,dx):
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
    levels = 6*placeUniformLevels(M)
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
M=64
#LM_levels = getNormalLM(M,500)
#print(LM_levels)
Unif_levels = 6*placeUniformLevels(M)
print(Unif_levels)
Clipped_levels = 3.29*placeUniformLevels(M)
print(Clipped_levels)
dx = 0.0001
qrange = np.arange(-6,6,dx)
pdf = norm.pdf(qrange,0,1)
Nsample=10000
#print(evaluate_MSE_monte_carlo(Nsample,LM_levels))
#print(evaluate_MSE_formula(LM_levels,pdf,qrange,dx))

print(evaluate_MSE_monte_carlo(Nsample,Unif_levels))
print(evaluate_MSE_formula(Unif_levels,pdf,qrange,dx))

print(evaluate_MSE_monte_carlo(Nsample,Clipped_levels))
print(evaluate_MSE_formula(Clipped_levels,pdf,qrange,dx))


tick_height=np.max(pdf)


#fig,ax=plt.subplots(figsize=(12,3))

#line0, = ax.plot(qrange,pdf,linewidth=2,color='k')
#line1 = ax.stem(LM_levels,tick_height*np.ones(M),basefmt=' ',markerfmt=' ',linefmt='r-')
#line2, = ax.plot(BADCS,MPC,linewidth=2,color='g',marker='^',markersize=10,label = 'MPC')
#plt.legend(handles=[line0,line1,line2],loc=0,fontsize=20)
#ax.grid()
#ax.set_xlabel(r'$x$',fontsize=20)
#ax.set_ylabel(r'$f_x$',fontsize=20)
#ax.set_xlim(0,16)
#ax.set_title('Lloyd-Max Quantizer',fontsize=30)
#ax.set_xticks(np.arange(1,20))
#ax.tick_params(axis='both',labelsize=15)
#plt.show()
#fig.savefig('LM_levels.svg',bbox_inches='tight')
#plt.clf()

fig,ax=plt.subplots(figsize=(12,3))

line0, = ax.plot(qrange,pdf,linewidth=2,color='k')
line1 = ax.stem(Unif_levels,tick_height*np.ones(M),basefmt=' ',markerfmt=' ',linefmt='r-')
#line2, = ax.plot(BADCS,MPC,linewidth=2,color='g',marker='^',markersize=10,label = 'MPC')
#plt.legend(handles=[line0,line1,line2],loc=0,fontsize=20)
ax.grid()
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$f_x$',fontsize=20)
#ax.set_xlim(0,16)
#ax.set_title('Lloyd-Max Quantizer',fontsize=30)
#ax.set_xticks(np.arange(1,20))
ax.tick_params(axis='both',labelsize=15)
plt.show()
fig.savefig('Unif_levels.svg',bbox_inches='tight')


plt.clf()

fig,ax=plt.subplots(figsize=(12,3))

line0, = ax.plot(qrange,pdf,linewidth=2,color='k')
line1 = ax.stem(Clipped_levels,tick_height*np.ones(M),basefmt=' ',markerfmt=' ',linefmt='r-')
#line2, = ax.plot(BADCS,MPC,linewidth=2,color='g',marker='^',markersize=10,label = 'MPC')
#plt.legend(handles=[line0,line1,line2],loc=0,fontsize=20)
ax.grid()
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$f_x$',fontsize=20)
#ax.set_xlim(0,16)
#ax.set_title('Lloyd-Max Quantizer',fontsize=30)
#ax.set_xticks(np.arange(1,20))
ax.tick_params(axis='both',labelsize=15)
plt.show()
fig.savefig('Clipped_levels.svg',bbox_inches='tight')