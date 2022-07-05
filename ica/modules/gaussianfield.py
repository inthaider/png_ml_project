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

import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
from scipy.signal.windows import general_hamming as hamming
from scipy.signal.windows import hann

import modules.fouriertransform as ft


# def power_array(Pk, k):
#     return np.where(k!=0, Pk(k), 0)
#     #return np.where(k==0, 0, Pk(k))

def dealiasx(f, kmaxknyq_ratio=(2/3)):
    """
    
    """

    N = f.size
    knyq = N//2
    kmax = int( kmaxknyq_ratio * knyq )

    fk = np.fft.rfft(f)
    k = np.fft.rfftfreq(N) * N

    khigh = np.ones(k.size) * kmax
    fk = np.where(k!=0, np.where(np.less_equal(k, khigh), fk, 0), fk)
    
    ff = np.fft.irfft(fk)

    return ff

def dealiask(N, fk, k, kmaxknyq_ratio=(2/3)):
    """
    
    """

    knyq = N//2
    kmax = int( kmaxknyq_ratio * knyq )

    khigh = np.ones(k.size) * kmax
    fk = np.where(k!=0, np.where(np.less_equal(k, khigh), fk, 0), fk)

    return fk

def gauss_var(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    u = np.random.uniform(size=size)
    e = 2 * np.pi * np.random.uniform(size=size)
    # a = np.sqrt(-2*np.log(u))
    a = np.sqrt(-np.log(u))
    
    return a * (np.cos(e) + 1j * np.sin(e))

def pk_primordial_1d(k, amp=1.0, ns=1.0):
    """

    TODO: Need to put into the GRF (g): tilt and amplitude 
    """
    
    pk = (np.pi / k) * amp * (k**(ns-1.0)) # Power spectrum

    return pk

# To generate 1D gaussian random field
def grf_zeta_1d(N, pk_amp=1.0, pk_ns=1.0, kmaxknyq_ratio=2/3, seed=None):
    """
    
    """

    grid = np.fft.rfftfreq(N) * N
    size = np.fft.rfftfreq(N).size
    # size = N//2+1 # Size of field is halved (floor division)
    # grid = np.arange(0, size) # 1D array with k-space positions

    grd = gauss_var(size, seed) # Gaussian random deviate in Fourier(k)-spae
    zk = np.zeros_like(grd)

    # g = np.random.normal(0, 1, size=size)
    # print(power_array_new(grid, 1))
    # print(np.sqrt(power_array_new(grid, 1)))

    zk[0] = 0
    zk[1:] = grd[1:] * np.sqrt( (2*np.pi / N) * pk_primordial_1d(grid[1:], pk_amp, pk_ns) )
    
    # plt.plot(grid, np.abs(zk))
    # plt.show()

    kmnr = kmaxknyq_ratio
    zk = dealiask(N, zk, grid, kmnr)

    # g =  np.where(grid!=0, g * np.sqrt(grid), 0) 

    out = np.fft.irfft(zk, N) # inverse FT --> out is GRF in real space
    s = np.std(out) # standard deviation
    m = np.mean(out)

    return (out - m) / s



def pk_chi_1d(k, amp, R, B=0.0):
    """

    """

    # 
    pk = (np.pi / k) * amp * (R * k)**2 * ( np.exp( -(R**2 * k**2) ) + B ) # Power spectrum
    
    return pk

def grf_chi_1d(N, pk_amp, pk_R, pk_B=0.0, kmaxknyq_ratio=2/3, seed=None):
    """

    """
    
    grid = np.fft.rfftfreq(N) * N
    size = np.fft.rfftfreq(N).size
    # size = N//2+1 # Size of field is halved (floor division)
    # grid = np.arange(0, size) # 1D array with k-space positionss

    grd = gauss_var(size, seed) # Gaussian random deviate in Fourier(k)-space
    ck = np.zeros_like(grd)

    # print(np.abs(grd[0]))

    ck[0] = 0
    ck[1:] = grd[1:] * np.sqrt( (2*np.pi / N) * pk_chi_1d(grid[1:], pk_amp, pk_R, pk_B))
    # ck = np.where(grid!=0, grd * np.sqrt( (2*np.pi / N) * pk_chi_1d(grid[1:], pk_amp, pk_R, pk_B))
    # ck = grd * np.sqrt( (2*np.pi / N) * pk_chi_1d(grid, pk_amp, pk_R, pk_B))

    kmnr = kmaxknyq_ratio
    ck = dealiask(N, ck, grid, kmnr)

    out = np.fft.irfft(ck, N) # inverse FT --> out is GRF in real space
    
    s = np.std(out) # standard deviation
    out = out / s
    m = np.mean(out)

    return out





def pk_primordial_los1d(k, amp=1.0, ns=1.0):
    """

    TODO: Need to put into the GRF (g): tilt and amplitude 
    """

    pk = np.where(k!=0, (2*np.pi**2) * (amp * (k**(ns-1.0))) / (k**3), 0) # Power spectrum

    return pk

# To generate 1D gaussian random field
def grf_zeta_los1d(N, pk_amp=1.0, pk_ns=1.0, seed=None):
    """
    
    """

    size = N//2+1 # Size of field is halved (floor division)
    grid = np.arange(0, size) # 1D array with k-space positions

    grd = gauss_var(size, seed) # Gaussian random deviate in Fourier(k)-spae
    zk = np.zeros_like(grd)

    # g = np.random.normal(0, 1, size=size)
    # print(power_array_new(grid, 1))
    # print(np.sqrt(power_array_new(grid, 1)))

    zk[0] = grd[0]
    zk[1:] = grd[1:] * np.sqrt(pk_primordial_los1d(grid[1:], pk_amp, pk_ns))

    # g =  np.where(grid!=0, g / np.sqrt(grid), 0) 

    out = np.fft.irfft(zk, N) # inverse FT --> out is GRF in real space
    s = np.std(out) # standard deviation

    return out / s



def pk_chi_los1d(k, amp, R, B=0.0):
    """

    """

    pk = np.where(k!=0, ( (2*np.pi**2) / k**3 ) * amp * (R * k)**2 * np.exp( (- R**2 * k**2) + B), 0) # Power spectrum
    
    return pk


def grf_chi_los1d(N, pk_amp, pk_R, pk_B=0.0, seed=None):
    """

    """
    
    size = N//2+1 # Size of field is halved (floor division)
    grid = np.arange(0, size) # 1D array with k-space positions
    grd = gauss_var(size, seed) # Gaussian random deviate in Fourier(k)-space
    ck = np.zeros_like(grd)

    grid = np.arange(0, size) # 1D array with k-space positions

    ck[0] = grd[0]
    ck[1:] = grd[1:] * np.sqrt(pk_chi_1d(grid[1:], pk_amp, pk_R, pk_B))

    out = np.fft.irfft(ck, N) # inverse FT --> out is GRF in real space
    s = np.std(out) # standard deviation

    return out / s








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

# Hamming window filter
def window_hamm(g, N, k_low, k_high):
    """Apply Hamming window filter in k-space.
    
    """
    x = np.fft.rfft(g)

    k = np.fft.rfftfreq(N) * N

    k_range = k_high - k_low
    hamm = np.zeros(np.shape(x))
    hamm[k_low:k_high] = hamming(k_range, 0.5)

    k_low = np.ones(np.shape(x))*k_low
    k_high = np.ones(np.shape(x))*k_high

    x = np.where(np.logical_and(np.less_equal(k, k_high), np.greater(k, k_low)), x, x*hamm)
    x_inv = np.fft.irfft(x)

    return x_inv, hamm

# Hann window filter
def window_hann(g, N, k_low, k_high):
    """Apply Hamming window filter in k-space.
    
    """
    x = np.fft.rfft(g)

    k = np.fft.rfftfreq(N) * N

    k_range = k_high - k_low
    filter_hann = np.zeros(np.shape(x))
    filter_hann[k_low:k_high] = hann(k_range, False)

    k_low = np.ones(np.shape(x))*k_low
    k_high = np.ones(np.shape(x))*k_high

    x = np.where(np.logical_and(np.less_equal(k, k_high), np.greater(k, k_low)), x, x*filter_hann)
    x_inv = np.fft.irfft(x)

    return x_inv, filter_hann




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

# # Gaussian bands
# def window_gauss(g, N, k_center, k_width):
#     x = np.fft.rfft(g)
#     k = np.fft.fftfreq(N) * N

#     W = np.exp(-1/2*(k-k_center)**2/(k_width)**2)
#     W_inv = np.fft.irfft(W, n=N)

#     x = x*W[:N//2+1]
#     x_inv = np.fft.irfft(x)

#     return x_inv, W_inv