import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from scipy.stats import binom
from scipy.special import comb #n choose k

def evaluate_snr(Yt,Yh):
    return 10.0*np.log10(np.var(Yt)/np.var(Yt-Yh))

def quantizeSigned(W,BW):
    Wq = np.minimum(np.round(W*np.power(2.0,BW-1.0))*np.power(2.0,1.0-BW),1.0-np.power(2.0,1.0-BW))
    return Wq
def quantizeUnsigned(X,BX):
    return np.minimum(np.round(X*np.power(2.0,BX))*np.power(2.0,-BX) ,1.0-np.power(2.0,-BX))

def extractBitsSigned(W,BW):
    W = np.minimum(W,1.0-np.power(2.0,-(BW-1.0)))
    Wbs = []
    Wbi = np.less(W,0).astype(float)
    Wbs.append(Wbi)
    W = (W + Wbi)
    for i in range(BW-1):
        Wbi = np.greater_equal(W,0.5).astype(float)
        Wbs.append(Wbi)
        W = 2.0*W - Wbi
    carry = np.greater_equal(W,0.5).astype(float)
    for i in range(BW):#-1):
        j=BW-1-i
        Wbs[j] = Wbs[j]+carry
        carry = np.greater(Wbs[j],1.5).astype(float)
        Wbs[j] = Wbs[j]*np.not_equal(Wbs[j],2.0)
    return Wbs

def extractBitsUnsigned(X,BX):
    X = np.minimum(X,1.0-np.power(2.0,-BX))
    Xbs = []
    for i in range(BX):
        Xbi = np.greater_equal(X,0.5).astype(float)
        Xbs.append(Xbi)
        X = 2.0*X - Xbi
    carry = np.greater_equal(X,0.5).astype(float)
    for i in range(BX):
        j=BX-1-i
        Xbs[j] = Xbs[j]+carry
        carry = np.greater(Xbs[j],1.5).astype(float)
        Xbs[j] = Xbs[j]*np.not_equal(Xbs[j],2.0)
    return Xbs

def reconstructSigned(Wbs,BW):
    W=np.zeros_like(Wbs[0])
    for j in range(BW):
        multiplier = np.power(0.5,j)
        if (j==0):
            multiplier=-1.0
        W+=Wbs[j]*multiplier
    return W

def reconstructUnsigned(Xbs,BX):
    X=np.zeros_like(Xbs[0])
    for l in range(BX):
        multiplier = np.power(0.5,l+1.0)
        X+=Xbs[l]*multiplier
    return X

def generate_random_data(Nsample, NDP):
    X = np.random.uniform(0,1,(NDP,Nsample))#np.clip(np.random.normal(0,1,(NDP,Nsample)),-1,1)#
    W = np.random.uniform(-1,1,(NDP,Nsample))#np.clip(np.random.normal(0,1,(NDP,Nsample)),-1,1)#
    return X,W,np.mean(X),np.var(X),np.mean(W),np.var(W)

def evaluate_binary_DP(Xb,Wb):
    return np.sum(Xb*Wb,axis=0)

def evaluate_DP(X,W):
    return np.sum(X*W,axis=0)

def quantize_adc_full_range(Yb,NDP,BADC):
    return NDP*quantizeUnsigned(Yb/NDP,BADC)


def quantize_adc_symmetric_clipping(Yb,NDP,BS,BADC):
    if BADC==2:
        zeta=1.71
    elif BADC==3:
        zeta=2.15
    elif BADC==4:
        zeta=2.55
    elif BADC==5:
        zeta=2.94
    elif BADC==6:
        zeta=3.29
    elif BADC==7:
        zeta=3.61
    elif BADC==8:
        zeta=3.98
    elif BADC==9:
        zeta=4.21
    else:
        zeta = 4.49
    mu = np.mean(Yb)#NDP*0.25*(np.power(2,BS)-1)#
    variance = np.var(Yb)#NDP*(np.power(2,BS)-1)*(5*np.power(2,BS)-1)/48.0#
    sigma = np.sqrt(variance)
    yL = mu-zeta*sigma#np.maximum(0,mu-zeta*sigma)
    yR = mu+zeta*sigma#np.minimum(NDP,mu+zeta*sigma)
    y_clipped = np.clip(Yb,yL,yR)
    DR = yR-yL
    y_quantized = yL+DR*quantizeUnsigned((y_clipped-yL)/DR,BADC)
    return y_quantized


def evaluate_sqnr(Nsample,NDP,BADC,BX,BW):
    SQNRs = []
    BSs = range(1,BX+1)
    X,W,muX,varX,muW,varW = generate_random_data(Nsample, NDP)
    Y = evaluate_DP(X,W)

    for BS in BSs:
        
        Xbs = extractBitsUnsigned(X,BX)
        Wbs = extractBitsSigned(W,BW)
        NFS = int(np.floor((BX+0.0)/BS))
        BI = BX-NFS*BS
        Ybw=[]
        for bw in range(BW):
            #first slice: 
            Wb = Wbs[bw]
            Yq = 0
            for i in range(1,NFS+1):
                Xi = np.power(2.0,BS)*reconstructUnsigned(Xbs[(i-1)*BS:i*BS],BS)
                Yi = evaluate_DP(Xi,Wb)
                Yiq = quantize_adc_symmetric_clipping(Yi,NDP,BS,BADC)
                Yq += (Yiq*np.power(2.0,-i*BS))
            # last slice
            if BI !=0:
                XIS = np.power(2.0,BI)*reconstructUnsigned(Xbs[-BI:],BI)
                YIS = evaluate_DP(XIS,Wb)
                YISq = quantize_adc_symmetric_clipping(YIS,NDP,BI,BADC)
                Yq += (YISq*np.power(2.0,1-(BX)))#subtract BI from BX
            Ybw.append(Yq)
        Ysliced = reconstructSigned(Ybw,BW)
        SQNRs.append(evaluate_snr(Y,Ysliced))

    return SQNRs

def predict_sqnr(NDP,BADC,BX,BW):
    SQNRs=[]
    X,W,muX,varX,muW,varW = generate_random_data(Nsample, NDP)
    EX2 = varX+np.square(muX)
    varY = NDP*EX2*varW#np.var(Y)#
    deltaX = np.power(2.0,-BX)
    deltaW = np.power(2.0,1-BW)
    sig2xy = np.square(deltaX)*NDP*varW/10.5
    sig2wy = np.square(deltaW)*NDP*EX2/10.5
    sigOCCs = [1.26e-1,3.79e-2,1.16e-2,3.5e-3,1.04e-3,3.04e-4,8.77e-5,2.49e-5,6.99e-6]
    sigLMs = [1.17e-1,3.45e-2,9.50e-3,2.5e-3,8.14e-4,2.13e-4,7.15e-5,2.02e-5,5.11e-6]
    zetas = [1.71,2.15,2.55,2.94,3.29,3.61,3.92,4.21,4.49]
    sigOCC = sigOCCs[BADC-2] #index zero when BADC=2
    sig2ADC_base = NDP*sigOCC*(1-np.power(4.0,-BX))*(1-np.power(4.0,-BW))/36.0
    BSs = range(1,BX+1)
    for BS in BSs:
        beta = np.true_divide(5.0-np.power(2.0,-BS),1.0+np.power(2.0,-BS))
        SQNR_linear = np.true_divide(varY,sig2xy+sig2wy+sig2ADC_base*beta)
        SQNRs.append(10*np.log10(SQNR_linear))
    return SQNRs


NDP=256
Nsample=1
BADC = 3
BXs= [8,10]#np.arange(4,9,2).tolist()
BW=4

fig,ax=plt.subplots(figsize=(9,6))
line_handles = []
#var_simulated_FR_all,var_simulated_SC_all,var_simulated_ICC_all,var_formula_FR_all,var_formula_SC_all,var_formula_ICC_all = evaluate_var_qadc(Nsample,NDP,BADCs,px,pws,zeta)
for BX in BXs[::-1]:
    SQNRs_predicted = predict_sqnr(NDP,BADC,BX,BW)
    SQNRs = evaluate_sqnr(Nsample,NDP,BADC,BX,BW)

    #line11, = ax.plot(BADCs,var_formula_FR_per_BADC,label=r'E: FR',linewidth=3,color = 'k')
    #line21, = ax.plot(pys,var_formula_SC_per_BADC,label=r'E: SC',linewidth=3,color = 'r')
    #line31, = ax.plot(pys,var_formula_ICC_per_BADC,label=r'E: ICC',linewidth=3,color='b')
    line11, = ax.plot(range(1,BX+1),SQNRs_predicted,label=r'E: $B_X$='+str(BX)+', $B_A$='+str(BADC),linewidth=2,marker='s',markersize=10)
    line1, = ax.plot(range(1,BX+1),SQNRs,label=r'S: $B_X$='+str(BX)+', $B_A$='+str(BADC),linewidth=2,linestyle='--',marker='s',markersize=10,color=line11.get_color())
    line_handles.append(line11)
    line_handles.append(line1)


BADC = 4
for BX in BXs[::-1]:
    SQNRs = evaluate_sqnr(Nsample,NDP,BADC,BX,BW)
    SQNRs_predicted = predict_sqnr(NDP,BADC,BX,BW)

    #line11, = ax.plot(BADCs,var_formula_FR_per_BADC,label=r'E: FR',linewidth=3,color = 'k')
    #line21, = ax.plot(pys,var_formula_SC_per_BADC,label=r'E: SC',linewidth=3,color = 'r')
    #line31, = ax.plot(pys,var_formula_ICC_per_BADC,label=r'E: ICC',linewidth=3,color='b')
    line11, = ax.plot(range(1,BX+1),SQNRs_predicted,label=r'E: $B_X$='+str(BX)+', $B_A$='+str(BADC),linewidth=2,marker='o',markersize=10)
    line1, = ax.plot(range(1,BX+1),SQNRs,label=r'S: $B_X$='+str(BX)+', $B_A$='+str(BADC),linewidth=2,linestyle='--',marker='o',markersize=10,color=line11.get_color())
    line_handles.append(line11)
    line_handles.append(line1)

BADC = 5
for BX in BXs[::-1]:
    SQNRs = evaluate_sqnr(Nsample,NDP,BADC,BX,BW)
    SQNRs_predicted = predict_sqnr(NDP,BADC,BX,BW)

    #line11, = ax.plot(BADCs,var_formula_FR_per_BADC,label=r'E: FR',linewidth=3,color = 'k')
    #line21, = ax.plot(pys,var_formula_SC_per_BADC,label=r'E: SC',linewidth=3,color = 'r')
    #line31, = ax.plot(pys,var_formula_ICC_per_BADC,label=r'E: ICC',linewidth=3,color='b')
    line11, = ax.plot(range(1,BX+1),SQNRs_predicted,label=r'E: $B_X$='+str(BX)+', $B_A$='+str(BADC),linewidth=2,marker='^',markersize=10)
    line1, = ax.plot(range(1,BX+1),SQNRs,label=r'S: $B_X$='+str(BX)+', $B_A$='+str(BADC),linewidth=2,linestyle='--',marker='^',markersize=10,color=line11.get_color())
    line_handles.append(line11)
    line_handles.append(line1)
plt.legend(handles=line_handles,bbox_to_anchor=(0, 2), loc='upper left',fontsize=20,ncol=6)
ax.set_xlabel(r'$B_S$',fontsize=30)
ax.set_ylabel(r'SQNR (dB)',fontsize=30)
#ax.set_yscale('log')
ax.tick_params(axis='both',labelsize=20)
ax.grid(axis='both',which='both')
#ax.set_ylim(bottom=0)
plt.savefig('SQNR_vs_BS.svg',bbox_inches='tight')


