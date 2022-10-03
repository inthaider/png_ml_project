"""
// Created by Jaafar.
Modified by Jibran Haider & Tom Morrison. //


**Contains the following functions**

"""


import numpy as np

# import pkg_resources
# import matplotlib.pyplot as plt
# from scipy.signal.windows import general_hamming as hamming
# from scipy.signal.windows import hann
# import modules.jaafar_fouriertransform as ft


############################################################
#
# Helper functions:
#
############################################################

# def power_array(Pk, k):
#     return np.where(k!=0, Pk(k), 0)
#     #return np.where(k==0, 0, Pk(k))

def gauss_var(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    u = np.random.uniform(size=size)
    e = 2 * np.pi * np.random.uniform(size=size)
    # a = np.sqrt(-2*np.log(u))
    a = np.sqrt(-np.log(u))
    
    return a * (np.cos(e) + 1j * np.sin(e))

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




############################################################
#
# Zeta GRFs:
#
############################################################

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



def pk_primordial_3d(k, amp=1.0, ns=1.0):
    """

    TODO: Need to put into the GRF (g): tilt and amplitude 
    """

    pk = np.where(k!=0, (2*np.pi**2) * (amp * (k**(ns-1.0))) / (k**3), 0) # Power spectrum

    return pk

# To generate 1D gaussian random field
def grf_zeta_3d_los1d(N, pk_amp=1.0, pk_ns=1.0, seed=None):
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
    zk[1:] = grd[1:] * np.sqrt(pk_primordial_3d(grid[1:], pk_amp, pk_ns))

    # g =  np.where(grid!=0, g / np.sqrt(grid), 0) 

    out = np.fft.irfft(zk, N) # inverse FT --> out is GRF in real space
    s = np.std(out) # standard deviation

    return out / s


############################################################
#
# Chi_e GRFS:
#
############################################################

def pk_chi_1d(k, amp, R, B=0.0):
    """ Power spectrum of 1D Chi_e.

    """

    # 
    pk = (np.pi / k) * amp * (R * k)**2 * ( np.exp( -(R**2 * k**2) ) + B ) # Power spectrum
    
    return pk

def grf_chi_1d(N, pk_amp, pk_R, pk_B=0.0, kmaxknyq_ratio=2/3, seed=None):
    """ Generate 1D Chi_e.

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




def pk_chi_3d(k, amp, R, B=0.0):
    """ Power spectrum of 3D Chi_e.

    """

    pk = np.where(k!=0, ( (2*np.pi**2) / k**3 ) * amp * (R * k)**2 * np.exp( (- R**2 * k**2) + B), 0) # Power spectrum
    
    return pk


def grf_chi_3d_los1d(N, pk_amp, pk_R, pk_B=0.0, seed=None):
    """ Generate 1D, 'Line-Of-Sight' Chi_e from 3D Chi_e.

    """
    
    size = N//2+1 # Size of field is halved (floor division)
    grid = np.arange(0, size) # 1D array with k-space positions
    grd = gauss_var(size, seed) # Gaussian random deviate in Fourier(k)-space
    ck = np.zeros_like(grd)

    grid = np.arange(0, size) # 1D array with k-space positions

    ck[0] = grd[0]
    ck[1:] = grd[1:] * np.sqrt(pk_chi_3d(grid[1:], pk_amp, pk_R, pk_B))

    out = np.fft.irfft(ck, N) # inverse FT --> out is GRF in real space
    s = np.std(out) # standard deviation

    return out / s