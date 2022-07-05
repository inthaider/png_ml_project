import modules.fouriertransform as ft
import numpy as np

def get_bins(num, grid):
    mx = np.max(grid)
    mn = np.min(grid)
    return np.linspace(mn, mx, num + 1)

def get_binweights(grid, numbins):
    bins = get_bins(numbins, grid)
    coord = grid.flatten()
    ind = np.digitize(coord, bins) 
    
    sumweights = np.bincount(ind, minlength=len(bins)+1)[1:-1]
    bins = np.bincount(ind, weights=coord, minlength=len(bins)+1)[1:-1] / sumweights
    
    return ind, bins, sumweights

def field_average(ind, x, sumweights, oneD=False):
    if oneD :
        return np.bincount(ind, weights=x, minlength=len(sumweights)+1)[1:-1] / sumweights
    else :
        return np.bincount(ind, weights=x.flatten(), minlength=len(sumweights)+1)[1:-1] / sumweights
    

def angular_average(x, kgrid, nbins, oneD=False):
    ind, bins, sumweights = get_binweights(kgrid, nbins)
    
    avg = field_average(ind, x, sumweights, oneD=oneD)
    
    return avg, bins

def power_spectrum(x1, x2=None, BoxSize=1.0, oneD=False):
    ft1, modes, grid = ft.fft(x1, BoxSize=BoxSize, modes=True, grid=True, oneD=oneD)
    
    if x2 is None:
        ft2 = ft1
    else: 
        ft2 = ft.fft(x2, BoxSize=BoxSize, oneD=oneD)

    N = x1.shape[0]
    pk = np.real(ft1 * np.conj(ft2))*(BoxSize**6/N**2)
    nbins = int(x1.shape[0] ** (2 / 2) / 2.1)
    P, kbins = angular_average(pk, grid, nbins, oneD=oneD)
    
    return P, kbins

def k_kurtosis(x, BoxSize=1.0, nbins=60):
    ft1, modes, grid = ft.fft(x, BoxSize=BoxSize, modes=True, grid=True)
    xk2 = np.abs(ft1)**2
    xk4, kbins = angular_average(xk2**2, grid, nbins)
    xk22 = angular_average(xk2, grid, nbins)[0]**2
    k4 = xk4/xk22-3
    return k4[kbins<=.5], kbins[kbins<=.5]

def k_skewness(x, BoxSize=1.0, nbins=60):
    ft1, modes, grid = ft.fft(x, BoxSize=BoxSize, modes=True, grid=True)
    xk = np.abs(ft1)
    xk = xk - np.mean(xk)
    xk3, kbins = angular_average(xk**3, grid, nbins)
    xk2 = angular_average(xk**2, grid, nbins)[0]**1.5
    sk = xk3/xk2
    return sk[kbins<=.5], kbins[kbins<=.5]


def cdf(x):
    X = np.sort(x.flatten())
    C = np.arange(len(X)) / (len(X) - 1)
    return X, C

def pdf(x):
    phi, C = cdf(x)
    n = len(phi)
    a = int(n**(1.7 / 3))
    phi_p = np.zeros(n + a)
    phi_p[:-a] = phi
    phi_m = np.zeros(n + a)
    phi_m[a:] = phi
    dphi = phi_p - phi_m
    return phi[:-a], a / ((n-1) * dphi[a:-a])
'''
def kfield_avg(x, nbins):
    N = len(x)
    xk, modes = ft.fft(x, modes=True)
    
    bins = np.linspace(-.5, (N-1)/(2*N), nbins)
    ind = np.digitize(modes, bins)
    w = 
''' 
def skewness(x):
    N = len(x)
    s3 = np.sum(x**3)/N**3
    s2 = np.sum(x**2)/N**3
    return s3/s2**1.5

def kurtosis(x):
    N = len(x)
    s4 = np.sum(x**4)/N**3
    s2 = np.sum(x**2)/N**3
    return s4/s2**2

#def negentropy(x):
    
