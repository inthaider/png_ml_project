import modules.fouriertransform as ft
import numpy as np

def gauss_var(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    u = np.random.uniform(size=size)
    e = 2 * np.pi * np.random.uniform(size=size)
    a = np.sqrt(-2*np.log(u))
    
    return a * (np.cos(e) + 1j * np.sin(e))

def power_array(Pk, k):
    return np.where(k!=0, Pk(k), 0)
    #return np.where(k==0, 0, Pk(k))

def gaussian_random_field(N, BoxSize=1.0, seed=None, Pk=lambda k: k**-3):
    size = (N, N, N)
    k_sq = ft.fftmodes(N)**2
    grid = np.sqrt(np.sum(np.meshgrid(k_sq, k_sq, k_sq), axis=0))
    g = gauss_var(size, seed)
    #g = np.random.normal(0, 1, size=size)
    g =  g * np.sqrt(power_array(Pk, grid))
    out = ft.ifft(g, BoxSize=BoxSize).real
    s = 1 #np.sqrt(np.sum(out**2) / N**3)
    return out / s
    
