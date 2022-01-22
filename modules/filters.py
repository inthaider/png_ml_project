import numpy as np
import modules.fouriertransform as ft
def filter_profile(x, y, z, s):
    
    #return (1 / ((2 * np.pi * s ** 2) ** 1.5 )) * np.exp(- .5 * (x**2 + y**2 + z**2) / s ** 2)
    d = np.sqrt(x**2 + y**2 + z**2)
    return np.where(d<=s, 3 / s**3, 0)

def ft_filter_profile(kx, ky, kz, s):
    return np.exp(- s**2 * (kx ** 2 + ky ** 2 + kz ** 2)/2)
    #kr = s * np.sqrt(kx**2 + ky**2 + kz**2)
    #return np.where(kr!=0, 3 * (np.sin(kr) - kr * np.cos(kr)) / kr**3, 0)

def filter_field(X, rf, BoxSize=1.0):
    ft1 = ft.fft(X, BoxSize=BoxSize)
    k = ft.fftmodes(X.shape[0], BoxSize=BoxSize)
    kx, ky, kz = np.meshgrid(k, k, k)
    #x, y, z = np.mgrid[:X.shape[0], :X.shape[0], :X.shape[0]]
    W = ft_filter_profile(kx, ky, kz, rf)
    #W = np.fft.fftn(filter_profile(x, y, z, rf))
    
    filtered_ft = ft1 * W
    filtered_X = ft.ifft(filtered_ft, BoxSize=BoxSize)
    return filtered_X
