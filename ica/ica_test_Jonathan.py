import numpy as np
import matplotlib.pyplot as plt

def center(x):
    xM = np.mean(x,axis=1,keepdims=True)
    return x - np.mean(x,axis=1,keepdims=True), xM

def covariance(x):
    x0 = center(x)[0]
    return (x0.dot(x0.T))/(x.shape[1]-1)

def whiten(x):
    cov = covariance(x)

    U,S,V = np.linalg.svd(cov)
    d = np.diag(1. / np.sqrt(S))
    whiten = np.dot(U,np.dot(d,U.T))
    xW = np.dot(whiten,x)
    return xW, whiten

def fastICA(signals, alpha=1, thresh=1.e-8, maxIt=100):
    """
    Assumes signal has had mean removed and has been prewhitened
    """
    m,n = signals.shape
    W = np.random.rand(m,m)

    for c in range(m):
        w = W[c,:].copy().reshape(m,1)
        w = w/np.sqrt(np.sum(w**2))

        i=0; lim = 10.*thresh
        while ((lim>thresh) & (i<maxIt)):
            ws = np.dot(w.T,signals)
            wg = np.tanh(ws*alpha).T
            wg_ = (1.-np.tanh(ws*alpha)**2)*alpha

            # Are the multiplications in here correct
            wNew = np.mean((signals * wg.T),axis=1)- wg_.mean() * w.squeeze()
            wNew = wNew - np.dot(np.dot(wNew,W[:c].T),W[:c])  # project previous solutions
            wNew = wNew / np.sqrt(np.sum(wNew**2))
            lim = np.abs(np.abs((wNew * w).sum()) - 1)
            
            w = wNew; i += 1
        W[c,:] = w.T
    return W

def extract_peak_and_wave(aPeak,aWave,wPeak=1.,wWave=1.,noise=np.random.normal):
    rv = np.linspace(-10.,10.,1000)

    fi = np.array([noise(size=rv.size).T,
                  aPeak*np.exp(-0.5*rv**2/wPeak**2),
                  aWave*np.sin(2.*np.pi*rv/wWave)] ).T
    # Mixing
    mix = np.array([[0.5,1,0.2],
                   [1.,0.5,0.4],
                   [0.5,0.8,1]])

    f = fi.dot(mix).T
    # Prewhiten and zero mean
    f0, fMean = center(f)
    fw, denoise = whiten(f0)

    w = fastICA(fw)
    decomp = fw.T.dot(w.T)
    decomp = (decomp.T-fMean).T
    
    return fi, decomp

def extract_peaks(num_samples, num_components, aPeak=[1.,1.],xPeak=[-5.,5.],wPeak=[1.,1.],noise=np.random.normal,ns=1000):
    rv = np.linspace(-10.,10.,ns)

    fi = np.vstack( [np.array([ac*np.exp(-0.5*(rv-xc)**2/wc**2) for ac,xc,wc in zip(aPeak,xPeak,wPeak)]),
                     noise(size=rv.size).T] ).T
    print(fi.shape)
    print(fi.shape[-1])
    n = fi.shape[-1]

    # Randomly mix signals
    # mix = (0.5+0.5*np.random.random(size=n**2)).reshape(n,n)
    mix = (0.5+0.5*np.random.random((num_components, num_samples)))
    
    f = fi.dot(mix).T
    f0, fMean = center(f)
    fw, denoise = whiten(f0)
    w = fastICA(fw)

    decomp = fw.T.dot(w.T)
    decomp = (decomp.T-fMean).T
    
    return fi, f, decomp

def extract_peaks_nn(num_samples, num_components, aPeak=[1.,1.],xPeak=[-5.,5.],wPeak=[1.,1.],noise=np.random.normal,ns=1000):
    rv = np.linspace(-10.,10.,ns)

    if noise == 0:
        fi = np.vstack( np.array([ac*np.exp(-0.5*(rv-xc)**2/wc**2) for ac,xc,wc in zip(aPeak,xPeak,wPeak)]) ).T
    else:
        fi = np.vstack( np.array([ac*np.exp(-0.5*(rv-xc)**2/wc**2) for ac,xc,wc in zip(aPeak,xPeak,wPeak)]) + 
                        noise(size=rv.size).T ).T

    # Randomly mix signals
    # mix = (0.5+0.5*np.random.random(size=n**2)).reshape(n,n)
    mix = (0.5+0.5*np.random.random((num_components, num_samples)))
    
    f = fi.dot(mix).T
    f0, fMean = center(f)
    fw, denoise = whiten(f0)
    w = fastICA(fw)

    decomp = fw.T.dot(w.T)
    decomp = (decomp.T-fMean).T
    
    return fi, f, decomp

def extract_peaks_nn_g(num_samples, num_components, aPeak=[1.,1.],xPeak=[-5.,5.],wPeak=[1.,1.],g=0,ns=1000):
    rv = np.linspace(-10.,10.,ns)

    fi = np.vstack( [np.array([ac*np.exp(-0.5*(rv-xc)**2/wc**2) for ac,xc,wc in zip(aPeak,xPeak,wPeak)]), g.T] ).T

    # Randomly mix signals
    # mix = (0.5+0.5*np.random.random(size=n**2)).reshape(n,n)
    mix = (0.5+0.5*np.random.random((num_components, num_samples)))
    
    f = fi.dot(mix).T
    f0, fMean = center(f)
    fw, denoise = whiten(f0)
    w = fastICA(fw)

    decomp = fw.T.dot(w.T)
    decomp = (decomp.T-fMean).T
    
    return fi, f, decomp

def plot_ica(s):
    f,a = plt.subplots()
    for i in range(s.shape[-1]):
        a.plot(s[:,i]+4.*i)
    return f,a

def kurtosis(s):
    """
    Compute kurtosis of a signal
    """
    mean = np.mean(s,axis=-2,keepdims=True)
    return np.mean((s-mean)**4,axis=-2)/np.std(s,axis=-2)**4-3.

if __name__=="__main__":
    #s,sr = extract_peak_and_wave(1.,1.)
    s,smix,sr = extract_peaks()
    
    f,a = plot_ica(sr)

    # Rank-order by a nonGaussianity measure (kurtosis for simplicity)
    nt = 100; a = np.empty((nt,s.shape[0],s.shape[1]))
    for i in range(nt):
        a[i] = extract_peaks(ns=1000)[-1]
    kur = kurtosis(a)
                        
