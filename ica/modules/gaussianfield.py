"""
// Created by Jaafar.
Modified by Jibran Haider & Tom Morrison. //


**Contains the following functions**

gauss_var(size, seed=None):
power_array(Pk, k):
power_array_new(k, amp, tilt=1):
gaussian_random_field(N, BoxSize=1.0, seed=None, Pk=lambda k: k**-3):
gaussian_random_field_1D(N, BoxSize=1.0, seed=None):
window(g, N, k_low, k_up):
window_gauss_log(g, N, k_center, k_width):
window_gauss_norm(g, N, k_center, k_width):
"""

import modules.fouriertransform as ft
import numpy as np

def gauss_var(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    u = np.random.uniform(size=size)
    e = 2 * np.pi * np.random.uniform(size=size)
    # a = np.sqrt(-2*np.log(u))
    a = np.sqrt(-np.log(u))
    
    return a * (np.cos(e) + 1j * np.sin(e))

def power_array(Pk, k):
    return np.where(k!=0, Pk(k), 0)
    #return np.where(k==0, 0, Pk(k))

def power_array_new(k, amp, tilt=1):

    return np.where(k!=0, amp*(k**(tilt-2)), 0)
    #return np.where(k==0, 0, Pk(k))

def gaussian_random_field(N, BoxSize=1.0, seed=None, Pk=lambda k: k**-3):
    size = (N, N, N)
    k_sq = ft.fftmodes(N)**2
    grid = np.sqrt(np.sum(np.meshgrid(k_sq, k_sq, k_sq), axis=0))
    g = gauss_var(size, seed)
    #g = np.random.normal(0, 1, size=size)
    g =  g #* np.sqrt(power_array(Pk, grid))
    out = ft.ifft(g, BoxSize=BoxSize).real
    s = np.sqrt(np.sum(out**2) / N**3)
    return out / s
    

# To generate 1D gaussian random field
def gaussian_random_field_1D(N, BoxSize=1.0, seed=None):
    
    size = N//2 # Size of field is halved (floor division)
    k_sq = ft.fftmodes(N)**2 # fftmodes returns the sample frequencies corresponding to the discrete FT
    grid = np.arange(0, size) # 1D array with real-space positions

    g = gauss_var(size, seed) # Gaussian random deviate in Fourier(k)-spae

    # g = np.random.normal(0, 1, size=size)
    # print(power_array_new(grid, 1))
    # print(np.sqrt(power_array_new(grid, 1)))

    g[0] = 0
    g[1:] = g[1:] / np.sqrt(grid[1:]) 
    """
    Need to put into the GRF (g): tilt and amplitude 
    """

    # g =  np.where(grid!=0, g / np.sqrt(grid), 0) 

    out = np.fft.irfft(g, N) # inverse FT --> out is GRF in real space
    s = np.std(out) # standard deviation

    return out / s

# Top hat bands
def window(g, N, k_low, k_up):
    x = np.fft.rfft(g)

    k = np.fft.fftfreq(N) * N
    # print(k[:N//2])

    k_low = np.ones(np.shape(x))*k_low
    k_up = np.ones(np.shape(x))*k_up

    x = np.where(np.logical_and(np.less_equal(k[:N//2+1], k_up), np.greater(k[:N//2+1], k_low)), x, 0)
    x_inv = np.fft.irfft(x)

    return x_inv

# # Gaussian bands
# def window_gauss(g, N, k_center, k_width):
#     x = np.fft.rfft(g)
#     k = np.fft.fftfreq(N) * N

#     W = np.exp(-1/2*(k-k_center)**2/(k_width)**2)
#     W_inv = np.fft.irfft(W, n=N)

#     x = x*W[:N//2+1]
#     x_inv = np.fft.irfft(x)

#     return x_inv, W_inv

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
