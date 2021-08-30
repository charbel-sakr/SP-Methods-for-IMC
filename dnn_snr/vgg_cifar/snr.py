import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from scipy.stats import binom
from scipy.special import comb #n choose k
from scipy.stats import norm
def evaluate_snr(Yt,Yh):
    #return 10.0*np.log10(np.var(Yt)/np.var(Yt-Yh))
    return 10.0*np.log10(np.sum(np.square(Yt))/np.sum(np.square(Yt-Yh)))

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

def evaluate_analog_DP(X,W,BS):
    X_max = np.power(2.0,BS)-1.0#np.max(X)#
    Co = 300e-15#femto
    Vdd = 1.0
    Vt = 0.4
    kappa = 0.08 #femto
    k=1.38e-23
    T=270
    sigma_cj = kappa*np.sqrt(Co*np.power(10.0,15))*np.power(10.0,-15)
    Vjs = Vdd*X*W/X_max
    sigma_theta = np.sqrt(k*T/Co)
    v_theta = np.random.normal(0,sigma_theta,Vjs.shape)
    pwl_cox = 0.5*0.31e-15
    v_inj = pwl_cox*(Vdd-Vt-Vjs)/Co
    Cjs = np.random.normal(Co,sigma_cj,Vjs.shape)
    numerator = np.sum(Cjs*(Vjs+v_theta+v_inj),axis=0)
    denominator = np.sum(Cjs,axis=0)

    return (NDP*X_max/Vdd)*numerator/denominator
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
    SQNR_simulated_OCC = []
    noise_factor = (1-np.power(4.0,-BW))*(1-np.power(4.0,-BX))*4.0/9.0

    X = np.transpose(np.load('activations.npy'))/2
    #print(np.min(X))
    #print(np.max(X))
    W = np.transpose(np.load('weights.npy'))
    #print(np.min(W))
    #print(np.max(W))
    muX = np.mean(X)
    varX = np.var(X)
    muW = np.mean(W)
    varW = np.var(W)

    EX2 = varX+np.square(muX)
    Y = evaluate_DP(X,W)
    #Y = evaluate_DP(quantizeUnsigned(X,7),quantizeSigned(W,7))
    Xbs = extractBitsUnsigned(X,BX)
    Wbs = extractBitsSigned(W,BW)

    varY = np.var(Y)#NDP*EX2*varW
    #deltaX = np.power(2.0,-BX)
    #deltaW = np.power(2.0,1-BW)
    #sig2xy = np.square(deltaX)*NDP*varW/10.5
    #sig2wy = np.square(deltaW)*NDP*EX2/10.5
    #sigOCCs = [1.26e-1,3.79e-2,1.16e-2,3.5e-3,1.04e-3,3.04e-4,8.77e-5,2.49e-5,6.99e-6]
    #sigLMs = [1.17e-1,3.45e-2,9.50e-3,2.5e-3,8.14e-4,2.13e-4,7.15e-5,2.02e-5,5.11e-6]
    #zetas = [1.71,2.15,2.55,2.94,3.29,3.61,3.92,4.21,4.49]
    zetas = [2.15,2.55,2.94,3.29,3.61,3.92,4.21,4.49,4.49]

    for (BADC,zeta_occ) in zip(BADCs,zetas):
        #print(BADC,zeta_occ)
        M=np.power(2.0,BADC).astype('int')
        #occ_levels = getNormalLM(M,100)
        #occ_levels = zeta_occ*placeUniformLevels(M)
        Ybx_FR = []
        Ybx_OCC = []
        for bx in range(BX):
            Ybw_FR = []
            Ybw_OCC = []
            for bw in range(BW):
                Yb = evaluate_analog_DP(Xbs[bx],Wbs[bw],1)
                Yb_FR = quantize_adc_full_range(Yb,NDP,BADC)
                #py = 0.25
                muYb = np.mean(Yb)#NDP*py#
                varYb = np.var(Yb)#NDP*py*(1-py)#
                zeta_right = np.minimum(NDP,np.round(muYb+zeta_occ*np.sqrt(varYb)))
                #zeta_right = np.minimum(NDP,(muYb+zeta_occ*np.sqrt(varYb)))
                zeta_left = np.maximum(0,np.round(muYb-zeta_occ*np.sqrt(varYb)))
                #zeta_left = np.maximum(0,(muYb-zeta_occ*np.sqrt(varYb)))
                if zeta_right==0:
                    Yb_OCC = Yb
                else:
                    delta_zeta= zeta_right-zeta_left
                    if delta_zeta>=M:
                        occ_levels = np.arange(zeta_left,zeta_right+1,delta_zeta/(M-1))
                    else:
                        occ_levels = np.arange(zeta_left-1,zeta_right+2)
                    #print(zeta_left,zeta_right,delta_zeta)
                    #occ_levels = np.arange(zeta_left,zeta_right+1,delta_zeta/(M-1))
                    Yb_OCC = custom_quantization(Yb,occ_levels)
                #Yb_OCC = muYb+np.sqrt(varYb)*custom_quantization((Yb-muYb)/np.sqrt(varYb),occ_levels)
                Ybw_FR.append(Yb_FR)
                Ybw_OCC.append(Yb_OCC)
            Ybx_FR.append(reconstructSigned(Ybw_FR,BW))
            Ybx_OCC.append(reconstructSigned(Ybw_OCC,BW))
        Y_FR = reconstructUnsigned(Ybx_FR,BX)
        Y_OCC = reconstructUnsigned(Ybx_OCC,BX)
        #Y = np.maximum(Y,0)
        #Y_FR = np.maximum(Y_FR,0)
        #Y_OCC = np.maximum(Y_OCC,0)
        SQNR_simulated_FR.append(evaluate_snr(Y,Y_FR))
        SQNR_simulated_OCC.append(evaluate_snr(Y,Y_OCC))


    return SQNR_simulated_FR,SQNR_simulated_OCC

def evaluate_sqnr_fs(Nsample,NDP,BADCs,BX,BW):
    SQNR_simulated_FR = []
    SQNR_simulated_OCC = []
    noise_factor = (1-np.power(4.0,-BW))*(1-np.power(4.0,-BX))*4.0/9.0

    X = np.transpose(np.load('activations.npy'))/2
    #print(np.min(X))
    #print(np.max(X))
    W = np.transpose(np.load('weights.npy'))
    #print(np.min(W))
    #print(np.max(W))
    muX = np.mean(X)
    varX = np.var(X)
    muW = np.mean(W)
    varW = np.var(W)

    EX2 = varX+np.square(muX)
    Y = evaluate_DP(X,W)
    #Y = evaluate_DP(quantizeUnsigned(X,7),quantizeSigned(W,7))
    Wbs = extractBitsSigned(W,BW)

    varY = np.var(Y)#NDP*EX2*varW
    #deltaX = np.power(2.0,-BX)
    #deltaW = np.power(2.0,1-BW)
    #sig2xy = np.square(deltaX)*NDP*varW/10.5
    #sig2wy = np.square(deltaW)*NDP*EX2/10.5
    #sigOCCs = [1.26e-1,3.79e-2,1.16e-2,3.5e-3,1.04e-3,3.04e-4,8.77e-5,2.49e-5,6.99e-6]
    #sigLMs = [1.17e-1,3.45e-2,9.50e-3,2.5e-3,8.14e-4,2.13e-4,7.15e-5,2.02e-5,5.11e-6]
    #zetas = [1.71,2.15,2.55,2.94,3.29,3.61,3.92,4.21,4.49]
    zetas = [2.15,2.55,2.94,3.29,3.61,3.92,4.21,4.49,4.49]
    Xq = quantizeUnsigned(X,BX)#*np.power(2.0,BX)
    #print(np.max(Xq),np.min(Xq))
    for (BADC,zeta_occ) in zip(BADCs,zetas):
        #print(BADC,zeta_occ)
        M=np.power(2.0,BADC).astype('int')
        #occ_levels = getNormalLM(M,100)
        #occ_levels = zeta_occ*placeUniformLevels(M)
        Ybw_FR = []
        Ybw_OCC = []
        for bw in range(BW):
            Yb = evaluate_analog_DP(Xq*np.power(2.0,BX),Wbs[bw],BX)/np.power(2.0,BX)
            #print(np.min(Yb),np.max(Yb))
            Yb_FR = quantize_adc_full_range(Yb,NDP,BADC)
            #py = 0.25
            Yb=Yb*np.power(2.0,BX)
            muYb = np.mean(Yb)#NDP*py#
            varYb = np.var(Yb)#NDP*py*(1-py)#
            zeta_right = np.minimum(np.max(Yb),np.round(muYb+zeta_occ*np.sqrt(varYb)))
            #zeta_right = np.minimum(NDP,(muYb+zeta_occ*np.sqrt(varYb)))
            zeta_left = np.maximum(np.min(Yb),np.round(muYb-zeta_occ*np.sqrt(varYb)))
            #zeta_left = np.maximum(0,(muYb-zeta_occ*np.sqrt(varYb)))
            if zeta_right==0:
                Yb_OCC = Yb
            else:
                delta_zeta= zeta_right-zeta_left
                if delta_zeta>=M:
                    #print(zeta_left,zeta_right,delta_zeta,M)
                    s = 0.15 if BADC>6 else 0.2 if BADC>5 else 0.68 if BADC ==5 else 0.8
                    occ_levels = np.arange(zeta_left,zeta_right,delta_zeta*s/M)
                    #print(occ_levels)
                    Yb_OCC = custom_quantization(Yb,occ_levels)
                else:
                    Yb_OCC=Yb#occ_levels = np.arange(zeta_left-1,zeta_right+2)
                #occ_levels = np.arange(zeta_left,zeta_right+1,delta_zeta/(M-1))
                
                #occ_levels = np.arange(zeta_left,zeta_right+1,delta_zeta/(M-1))
                #Yb_OCC = custom_quantization(Yb,occ_levels)
            #Yb_OCC = muYb+np.sqrt(varYb)*custom_quantization((Yb-muYb)/np.sqrt(varYb),occ_levels)
            Ybw_FR.append(Yb_FR)
            Ybw_OCC.append(Yb_OCC)
        Y_FR = reconstructSigned(Ybw_FR,BW)
        Y_OCC = reconstructSigned(Ybw_OCC,BW)/np.power(2.0,BX)
        #Y = np.maximum(Y,0)
        #Y_FR = np.maximum(Y_FR,0)
        #Y_OCC = np.maximum(Y_OCC,0)
        SQNR_simulated_FR.append(evaluate_snr(Y,Y_FR))
        SQNR_simulated_OCC.append(evaluate_snr(Y,Y_OCC))


    return SQNR_simulated_FR,SQNR_simulated_OCC
        

NDP=256
Nsample=4000
BADCs = np.arange(2,11).tolist()
BX=6
BW=6


#var_simulated_FR_all,var_simulated_SC_all,var_simulated_ICC_all,var_formula_FR_all,var_formula_SC_all,var_formula_ICC_all = evaluate_var_qadc(Nsample,NDP,BADCs,px,pws,zeta)
SQNR_bsbp_FR,SQNR_bsbp_OCC = evaluate_sqnr(Nsample,NDP,BADCs,BX,BW)
SQNR_iswp_FR,SQNR_iswp_OCC = evaluate_sqnr_fs(Nsample,NDP,BADCs,BX,BW)
print(SQNR_bsbp_FR)
print(SQNR_bsbp_OCC)
print(SQNR_iswp_FR)
print(SQNR_iswp_OCC)
SQNR_MAX = SQNR_bsbp_FR[-3]
fig,ax=plt.subplots(figsize=(9,6))
line_handles = []
line1, = ax.plot(BADCs,SQNR_bsbp_FR,label=r'(1,FR)',linewidth=3,linestyle='--',color = 'k',marker = '^',markersize=15)
line2, = ax.plot(BADCs,SQNR_bsbp_OCC,label=r'(1,OCC)',linewidth=3,linestyle='--',color = 'r',marker = 'o',markersize=15)
#line11, = ax.plot(BADCs,SQNR_iswp_FR,label=r'(BX,FR)',linewidth=3,linestyle='--',color = 'b',marker = '^',markersize=15)
line22, = ax.plot(BADCs,SQNR_iswp_OCC,label=r'($B_X$,OCC)',linewidth=3,linestyle='--',color = 'g',marker = 'o',markersize=15)
line3 = ax.plot([1,11],[SQNR_MAX,SQNR_MAX],linewidth=3,color='k',linestyle='dotted')
line_handles.append(line1)
#line_handles.append(line11)
line_handles.append(line2)
line_handles.append(line22)
plt.legend(handles=line_handles,loc=0,fontsize=20,ncol=1)
ax.set_xlabel(r'$B_A$',fontsize=30)
ax.set_ylabel(r'SNR (dB)',fontsize=30)
#ax.set_yscale('log')
ax.tick_params(axis='both',labelsize=20)
ax.grid(axis='both',which='both')
ax.set_ylim(bottom=0)
ax.set_xlim(left=2,right=10)
plt.savefig('SNR_vs_BADC_occ.svg',bbox_inches='tight')

BADCs = np.arange(2,11)
cap = 30e-15
k1=1e-13
k2=1e-18
EA = k1*BADCs+k2*np.power(4.0,BADCs)
EQR = 0.05*cap

EOP_BS = BX*(EQR+EA/NDP)
EOP_FS = EQR+EA/NDP


print('got here')

fig,ax=plt.subplots(figsize=(9,6))
line_handles = []
line1, = ax.plot(1e15*EOP_BS,SQNR_bsbp_FR,label=r'(1,FR)',linewidth=3,linestyle='--',color = 'k',marker = '^',markersize=15)
line2, = ax.plot(1e15*EOP_BS,SQNR_bsbp_OCC,label=r'(1,OCC)',linewidth=3,linestyle='--',color = 'r',marker = 'o',markersize=15)
#line11, = ax.plot(1e15*EOP_FS,SQNR_iswp_FR,label=r'(BX,FR)',linewidth=3,linestyle='--',color = 'b',marker = '^',markersize=15)
line22, = ax.plot(1e15*EOP_FS,SQNR_iswp_OCC,label=r'($B_X$,OCC)',linewidth=3,linestyle='--',color = 'g',marker = 'o',markersize=15)
line_handles.append(line1)
#line_handles.append(line11)
line_handles.append(line2)
line_handles.append(line22)
plt.legend(handles=line_handles,loc='lower right', bbox_to_anchor=(1,0),fontsize=20,ncol=1)
ax.set_xlabel(r'$E_{OP}$ (fJ)',fontsize=30)
ax.set_ylabel(r'SNR (dB)',fontsize=30)
#ax.set_xscale('log')
ax.tick_params(axis='both',labelsize=20)
ax.grid(axis='both',which='both')
ax.set_ylim(bottom=0)
ax.set_xlim(right=32,left=2)
plt.savefig('SNR_vs_eop_occ.svg',bbox_inches='tight')
