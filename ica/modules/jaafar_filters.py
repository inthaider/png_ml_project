"""
// Created by Jaafar.
Modified by Jibran Haider. //

"""

import numpy as np
import jaafar_fouriertransform as ft

def window_tophat(f, kmin, kmax):
    """
    
    """
    
    N = f.size
    fk = np.fft.rfft(f)
    k = np.fft.rfftfreq(N) * N

    klow = np.ones(np.shape(fk))*kmin
    kup = np.ones(np.shape(fk))*kmax

    fk = np.where(k!=0, (np.where(np.logical_and(np.less_equal(k, kup), np.greater_equal(k, klow)), fk, 0)), fk)
    
    ff = np.fft.irfft(fk)

    return ff

# Gaussian bands with log spacing
def window_gauss_log(g, N, k_center, k_width):
    x = np.fft.rfft(g)
    k = np.fft.fftfreq(N) * N

    W = np.where(k != 0, np.exp(-1/2*(np.log(k)-np.log(k_center))**2/(np.log(k_width))**2), 0)
    W_inv = np.fft.irfft(W, n=N)

    x = x*W[:N//2+1]
    x_inv = np.fft.irfft(x)

    return x_inv, W_inv

# Normalized gaussian bands
def window_gauss_norm(g, N, k_center, k_width):
    x = np.fft.rfft(g)
    k = np.fft.fftfreq(N) * N
    # print(k)

    W = np.exp(-1/2*(k-k_center)**2/(k_width)**2)
    W_inv = np.fft.irfft(W, n=N)
    norm = np.sum(W_inv) # put in normalization for box size (box size / N)

    x = x*W[:N//2+1]
    x_inv = np.fft.irfft(x)/norm

    return x_inv, W_inv/norm

# Gaussian bands
def window_gauss(g, N, k_center, k_width):
    x = np.fft.rfft(g)
    k = np.fft.fftfreq(N) * N

    W = np.exp(-1/2*(k-k_center)**2/(k_width)**2)
    W_inv = np.fft.irfft(W, n=N)

    x = x*W[:N//2+1]
    x_inv = np.fft.irfft(x)

    return x_inv, W_inv


############################################################
#
# JAAFAR'S STUFF
#
############################################################


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