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
    '''It generates a complex Gaussian random variable with zero mean and unit variance
    
    Parameters
    ----------
    size
        the size of the output array
    seed
        the seed for the random number generator. If None, then the random number generator is not seeded.
    
    Returns
    -------
        a complex number.
    
    '''
    if seed is not None:
        np.random.seed(seed)
    
    u = np.random.uniform(size=size)
    e = 2 * np.pi * np.random.uniform(size=size)
    # a = np.sqrt(-2*np.log(u))
    a = np.sqrt(-np.log(u))
    
    return a * (np.cos(e) + 1j * np.sin(e))

def dealiasx(f, kmaxknyq_ratio=(2/3)):
    '''> If the wavenumber is less than or equal to the maximum wavenumber, keep the Fourier coefficient.
    Otherwise, set it to zero
    
    Parameters
    ----------
    f
        the input array
    kmaxknyq_ratio
        The ratio of the maximum wavenumber to the Nyquist wavenumber.
    
    Returns
    -------
        the inverse Fourier transform of the input array.
    
    '''

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
    '''If the wavenumber is less than or equal to the maximum wavenumber, then keep the value of the
    Fourier transform. Otherwise, set it to zero
    
    Parameters
    ----------
    N
        the number of points in the signal
    fk
        the fourier transform of the image
    k
        the wavenumber array
    kmaxknyq_ratio
        the ratio of the maximum wavenumber to the Nyquist wavenumber.
    
    Returns
    -------
        the Fourier transform of the input signal.
    
    '''

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

def pk_primordial_1d(k, amplitude=1.0, n_s=1.0):
    '''It returns the primordial power spectrum in 1D

    This is a function that returns the primordial power spectrum in 1D.
    It is a one-parameter model of the primordial power spectrum.
    It is a power law with a scale-invariant power spectrum.
    The power law is of the form P(k) = (k^ns - 1.0)
    
    Parameters
    ----------
    k
        The wavenumber.
    amplitude
        This is the amplitude of the power spectrum.
    n_s
        The spectral index of the primordial power spectrum.
    
    Returns
    -------
        The power spectrum.
    
    TODO: Need to put into the GRF (g): tilt and amplitude 
    '''
    if np.isscalar(k):
        k = np.array([k])

    if np.any(k < 0):
        raise ValueError("k must be greater than 0.")

    power_spectrum = (np.pi / k) * amplitude * (k**(n_s-1.0)) # Power spectrum

    return power_spectrum

def grf_zeta_1d(N, pk_amp=1.0, pk_ns=1.0, kmaxknyq_ratio=2/3, seed=None):
    """Generate a 1D Gaussian random field with power spectrum given by
    the primordial power spectrum.

    Parameters
    ----------
    N : int
        Size of the input array.
    pk_amp : float
        Amplitude of the primordial power spectrum.
    pk_ns : float
        Spectral index of the primordial power spectrum.
    kmaxknyq_ratio : float
        Ratio of kmax to the Nyquist frequency. The Nyquist frequency is
        given by np.pi * N. The default value of 2/3 is consistent with
        the usual practice of simulating a 2D field with a 1D FFT.
    seed : int, optional
        Seed for the random number generator. If set to None, the seed is
        set to 0.

    Returns
    -------
    (out - m) / s : ndarray
        Gaussian random field in real space.

    """
    if not np.isfinite(N):
        raise ValueError("N must be finite.")
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")
    if not np.isfinite(pk_amp):
        raise ValueError("pk_amp must be finite.")
    if pk_amp <= 0:
        raise ValueError("pk_amp must be positive.")
    if not np.isfinite(pk_ns):
        raise ValueError("pk_ns must be finite.")
    if pk_ns <= -2:
        raise ValueError("pk_ns must be larger than -2.")
    if not np.isfinite(kmaxknyq_ratio):
        raise ValueError("kmaxknyq_ratio must be finite.")
    if kmaxknyq_ratio < 0:
        raise ValueError("kmaxknyq_ratio must be non-negative.")
    if not np.isfinite(seed):
        raise ValueError("seed must be finite.")
    if seed is not None:
        seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    grid = np.fft.rfftfreq(N) * N
    size = np.fft.rfftfreq(N).size

    grd = gauss_var(size, seed)
    zk = np.zeros_like(grd)

    zk[0] = 0
    zk[1:] = grd[1:] * np.sqrt( (2*np.pi / N) * pk_primordial_1d(grid[1:], pk_amp, pk_ns) )
    
    kmnr = kmaxknyq_ratio
    zk = dealiask(N, zk, grid, kmnr)

    out = np.fft.irfft(zk, N)
    s = np.std(out)
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