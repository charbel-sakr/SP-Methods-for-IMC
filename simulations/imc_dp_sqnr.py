import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from scipy.stats import binom
from scipy.special import comb #n choose k
from scipy.stats import norm
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


def quantize_adc_symmetric_clipping(Yb,NDP,py,BADC,zeta):
    mu = NDP*py
    sigma = np.sqrt(NDP*py*(1-py))
    yL = np.maximum(0,mu-zeta*sigma)
    yR = np.minimum(NDP,mu+zeta*sigma)
    y_clipped = np.clip(Yb,yL,yR)
    DR = yR-yL
    y_quantized = yL+DR*quantizeUnsigned((y_clipped-yL)/DR,BADC)
    return y_quantized,yL,yR


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

def custom_quantization(data,quantizationLevels):
    #data = np.random.triangular(-1,0,1,Nsample)
    Nsample = data.shape[0]
    quantizedData = np.zeros(Nsample)
    for i in range(Nsample):
        quantizedData[i]=quantizationLevels[np.argmin(np.abs(data[i]-quantizationLevels))]
    #SNR = 10*np.log10(np.true_divide(np.sum(np.var(data)),np.sum(np.var(data-quantizedData))))
    
    return quantizedData

def evaluate_sqnr(Nsample,NDP,BADCs,BX,BW):
    SQNR_simulated_FR = []
    SQNR_simulated_LM = []
    SQNR_simulated_OCC = []
    noise_factor = (1-np.power(4.0,-BW))*(1-np.power(4.0,-BX))*4.0/9.0

    SQNR_formula_FR = []
    SQNR_formula_OCC = []
    SQNR_formula_LM = []

    X,W,muX,varX,muW,varW = generate_random_data(Nsample, NDP)
    EX2 = varX+np.square(muX)
    Y = evaluate_DP(X,W)
    Xbs = extractBitsUnsigned(X,BX)
    Wbs = extractBitsSigned(W,BW)

    varY = np.var(Y)#NDP*EX2*varW
    deltaX = np.power(2.0,-BX)
    deltaW = np.power(2.0,1-BW)
    sig2xy = np.square(deltaX)*NDP*varW/10.5
    sig2wy = np.square(deltaW)*NDP*EX2/10.5
    sigOCCs = [1.26e-1,3.79e-2,1.16e-2,3.5e-3,1.04e-3,3.04e-4,8.77e-5,2.49e-5,6.99e-6]
    sigLMs = [1.17e-1,3.45e-2,9.50e-3,2.5e-3,8.14e-4,2.13e-4,7.15e-5,2.02e-5,5.11e-6]
    zetas = [1.71,2.15,2.55,2.94,3.29,3.61,3.92,4.21,4.49]

    for (BADC,zeta_occ,sigOCC,sigLM) in zip(BADCs,zetas,sigOCCs,sigLMs):
        M=np.power(2.0,BADC).astype('int')
        LM_levels = getNormalLM(M,100)
        occ_levels = zeta_occ*placeUniformLevels(M)
        Ybx_FR = []
        Ybx_OCC = []
        Ybx_LM = []
        for bx in range(BX):
            Ybw_FR = []
            Ybw_OCC = []
            Ybw_LM = []
            for bw in range(BW):
                Yb = evaluate_binary_DP(Xbs[bx],Wbs[bw])
                Yb_FR = quantize_adc_full_range(Yb,NDP,BADC)
                #py = 0.25
                muYb = np.mean(Yb)#NDP*py
                varYb = np.var(Yb)#NDP*py*(1-py)
                Yb_OCC = muYb+np.sqrt(varYb)*custom_quantization((Yb-muYb)/np.sqrt(varYb),occ_levels)
                Yb_LM = muYb+np.sqrt(varYb)*custom_quantization((Yb-muYb)/np.sqrt(varYb),LM_levels)
                Ybw_FR.append(Yb_FR)
                Ybw_OCC.append(Yb_OCC)
                Ybw_LM.append(Yb_LM)
            Ybx_FR.append(reconstructSigned(Ybw_FR,BW))
            Ybx_OCC.append(reconstructSigned(Ybw_OCC,BW))
            Ybx_LM.append(reconstructSigned(Ybw_LM,BW))
        Y_FR = reconstructUnsigned(Ybx_FR,BX)
        Y_OCC = reconstructUnsigned(Ybx_OCC,BX)
        Y_LM = reconstructUnsigned(Ybx_LM,BX)
        SQNR_simulated_FR.append(evaluate_snr(Y,Y_FR))
        SQNR_simulated_OCC.append(evaluate_snr(Y,Y_OCC))
        SQNR_simulated_LM.append(evaluate_snr(Y,Y_LM))

        deltaY = NDP*np.power(2.0,-BADC)
        sigqY = noise_factor*np.square(deltaY)/12.0
        slice_factor = noise_factor*NDP*0.25*0.75
        SQNR_formula_FR.append(10*np.log10(varY/(sig2wy+sig2xy+sigqY)))
        SQNR_formula_LM.append(10*np.log10(varY/(sig2wy+sig2xy+slice_factor*sigLM)))
        SQNR_formula_OCC.append(10*np.log10(varY/(sig2wy+sig2xy+slice_factor*sigOCC)))

    return SQNR_simulated_FR,SQNR_simulated_OCC,SQNR_simulated_LM,SQNR_formula_FR,SQNR_formula_OCC,SQNR_formula_LM
        

NDP=256
Nsample=1000
BADCs = np.arange(2,11).tolist()
BX=4
BW=4


#var_simulated_FR_all,var_simulated_SC_all,var_simulated_ICC_all,var_formula_FR_all,var_formula_SC_all,var_formula_ICC_all = evaluate_var_qadc(Nsample,NDP,BADCs,px,pws,zeta)
SQNR_simulated_FR,SQNR_simulated_OCC,SQNR_simulated_LM,SQNR_formula_FR,SQNR_formula_OCC,SQNR_formula_LM = evaluate_sqnr(Nsample,NDP,BADCs,BX,BW)
fig,ax=plt.subplots(figsize=(9,6))
line_handles = []
line11, = ax.plot(BADCs,SQNR_formula_FR,label=r'E: FR',linewidth=3,color = 'k',marker = '^',markersize=15)
line31, = ax.plot(BADCs,SQNR_formula_LM,label=r'E: LM',linewidth=3,color='b',marker = 's',markersize=15)
line21, = ax.plot(BADCs,SQNR_formula_OCC,label=r'E: OCC',linewidth=3,color = 'r',marker = 'o',markersize=15)
line1, = ax.plot(BADCs,SQNR_simulated_FR,label=r'S: FR',linewidth=3,linestyle='--',color = 'k',marker = '^',markersize=15)
line3, = ax.plot(BADCs,SQNR_simulated_LM,label=r'S: LM',linewidth=3,linestyle='--',color='b',marker = 's',markersize=15)
line2, = ax.plot(BADCs,SQNR_simulated_OCC,label=r'S: OCC',linewidth=3,linestyle='--',color = 'r',marker = 'o',markersize=15)
line_handles.append(line11)
line_handles.append(line1)
line_handles.append(line21)
line_handles.append(line2)
line_handles.append(line31)
line_handles.append(line3)
plt.legend(handles=line_handles,loc=0,fontsize=20,ncol=1)
ax.set_xlabel(r'$B_A$',fontsize=30)
ax.set_ylabel(r'SQNR (dB)',fontsize=30)
#ax.set_yscale('log')
ax.tick_params(axis='both',labelsize=20)
ax.grid(axis='both',which='both')
ax.set_ylim(bottom=0)
plt.savefig('SQNR_vs_BADC_occ.svg',bbox_inches='tight')