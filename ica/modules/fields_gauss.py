# 
# Created by Jaafar.
# Modified by Jibran Haider & Tom Morrison.
# 
"""This module contains functions for generating initial conditions for the 1D \zeta and \chi_e^2 Gaussian Random Fields (GRFs).

Routine Listings
----------------
pk_primordial_1d
    Returns the primordial power spectrum for a pure 1D \zeta GRF.
grf_zeta_1d
    Generate a 1D \zeta GRF with power spectrum given by the primordial power spectrum.

pk_primordial_3d
    Returns the primordial power spectrum for a pure 3D \zeta GRF.
grf_zeta_3d_los1d
    Generate a 1D \zeta GRF from the power spectrum for a 3D
    \zeta GRF (acts as a line-of-sight 1D strip).

pk_chi_1d
    Returns the power spectrum for a 1D \chi_e^2 GRF.
grf_chi_1d
    Generate a 1D \chi_e^2 GRF from the power spectrum for a 1D \chi_e^2 GRF.
    
pk_chi_3d
    Returns the power spectrum for a 3D \chi_e^2 GRF.
grf_chi_3d_los1d
    Generate a 1D, 'Line-Of-Sight' \chi_e^2 GRF from the power spectrum 
    generated for a 3D \chi_e^2 GRF.

gauss_var
    Generates a complex Gaussian random variable in Fourier space with zero mean and unit variance.
dealiasx
    Dealias a real space field and return the de-aliased field in real space domain.
dealiask
    Dealias a Fourier space field and return the de-aliased field in Fourier space domain.
    
TODO
----
Change the numpy random seed generation to the new, recommended method.
"""

import numpy as np
# import pkg_resources
# import matplotlib.pyplot as plt
# from scipy.signal.windows import general_hamming as hamming
# from scipy.signal.windows import hann
# import modules.jaafar_fouriertransform as ft


############################################################
#
# 1D \zeta GRF:
#
############################################################
def pk_primordial_1d(k, amplitude=1.0, n_s=1.0):
    """Returns the primordial power spectrum for a pure 1D \zeta GRF.

    This is a function that returns the primordial power spectrum in 1D.
    It is a one-parameter model of the primordial power spectrum.
    It is a power law with a scale-invariant power spectrum.
    The power law is of the form P(k) = (pi/k) * A * k^(n_s-1), where 
    k is the wavenumber, A is the amplitude, and n_s is the spectral index.
    
    Parameters
    ----------
    k
        Array of wavenumbers.
    amplitude : float, optional
        Amplitude of the primordial power spectrum.
    n_s : float, optional
        Spectral index of the primordial power spectrum.
    
    Returns
    -------
    power_spectrum
        The primordial power spectrum.
    
    TODO
    ----
    Need to put into the GRF (g): tilt and amplitude 
    """
    if np.isscalar(k):
        k = np.array([k])

    if np.any(k < 0):
        raise ValueError("k must be greater than 0.")

    power_spectrum = (np.pi / k) * amplitude * (k**(n_s-1.0)) # Power spectrum

    return power_spectrum

def grf_zeta_1d(N, pk_amp=1.0, pk_ns=1.0, kmaxknyq_ratio=2/3, seed=None):
    """Generate a 1D \zeta GRF with power spectrum given by
    the primordial power spectrum.

    Parameters
    ----------
    N : int
        Size of the input array.
    pk_amp : float, optional
        Amplitude of the primordial power spectrum.
    pk_ns : float, optional
        Spectral index of the primordial power spectrum.
    kmaxknyq_ratio : float, optional
        Ratio of kmax to the Nyquist frequency. The Nyquist frequency is
        given by np.pi * N. The default value of 2/3 is consistent with
        the usual practice of simulating a 2D field with a 1D FFT.
    seed : optional
        Seed for the random number generator. If set to None, the seed is
        set to 0. If set to an integer, it is used as the seed directly.
        If set to a tuple, it is assumed to be a random state generated by
        np.random.get_state().

    Returns
    -------
    zeta
        The mean-subtracted, variance-normalized \zeta GRF.

    Notes
    -----

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

    # If seed is provided as a tuple, use it to set the state of the random number generator
    if isinstance(seed, tuple):
        np.random.set_state(seed)
    # If seed is provided as an integer, use it to seed the random number generator
    elif isinstance(seed, int):
        np.random.seed(seed)
    # If seed is not provided, use the default seed value of 0
    elif seed is None:
        np.random.seed(0)
    else:
        raise ValueError("Invalid seed value: seed must be an integer or a tuple.")

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


############################################################
#
# 1D Line-of-Sight \zeta GRF from P_k for 3D GRF:
#
############################################################
def pk_primordial_3d(k, amp=1.0, ns=1.0):
    """Returns the primordial power spectrum for a 3D \zeta GRF.

    Parameters
    ----------
    k
        Array of wavenumbers.
    amp : float, optional
        Amplitude of the primordial power spectrum.
    ns : float, optional
        Spectral index of the primordial power spectrum.

    Returns
    -------
    pk
        The primordial power spectrum in 3D.

    Notes
    -----
    The power spectrum is given by P(k) = (2*pi^2) * A * k^(n_s-1) / k^3.

    TODO
    ----
    Need to put into the GRF (g): tilt and amplitude 
    """

    pk = np.where(k!=0, (2*np.pi**2) * (amp * (k**(ns-1.0))) / (k**3), 0) # Power spectrum

    return pk

# To generate 1D \zeta GRF
def grf_zeta_3d_los1d(N, pk_amp=1.0, pk_ns=1.0, seed=None):
    """Generate a 1D \zeta GRF from the power spectrum for a 3D
    \zeta GRF (acts as a line-of-sight 1D strip).

    Parameters
    ----------
    N : int
        Size of the real space field.
    pk_amp : float, optional
        Amplitude of the primordial power spectrum.
    pk_ns : float, optional
        Spectral index of the primordial power spectrum.
    seed : optional
        Seed for the random number generator. If set to None, the seed is
        set to 0. If set to an integer, it is used as the seed directly.
        If set to a tuple, it is assumed to be a random state generated by
        np.random.get_state().

    Returns
    -------
    zeta
        The variance-normalized \zeta GRF.        
    
    Notes
    -----

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
# 1D \chi_e^2 Field:
#
############################################################
def pk_chi_1d(k, amp, R, B=0.0):
    r"""Returns the power spectrum for a 1D \chi_e^2 GRF.

    Parameters
    ----------
    k
        Array of wavenumbers.
    amp
        Amplitude of the primordial power spectrum.
    R
        Parameter for the power spectrum.
    B : float, optional
        Parameter for the power spectrum.

    Returns
    -------
    pk
        The power spectrum in 1D.

    Notes
    -----
    The power spectrum is given by P(k) = (pi / k) * A * R^2 * k^2 * (exp(-R^2 * k^2) + B).
    """

    # 
    pk = (np.pi / k) * amp * (R * k)**2 * ( np.exp( -(R**2 * k**2) ) + B ) # Power spectrum
    
    return pk

def grf_chi_1d(N, pk_amp, pk_R, pk_B=0.0, kmaxknyq_ratio=2/3, seed=None):
    r"""Generate a 1D \chi_e^2 GRF from the power spectrum.

    Parameters
    ----------
    N : int
        Size of the real space field.
    pk_amp
        Amplitude of the primordial power spectrum.
    pk_R
        Parameter for the power spectrum.
    pk_B : float, optional
        Parameter for the power spectrum.
    kmaxknyq_ratio : float, optional
        Ratio of the maximum wavenumber to the Nyquist wavenumber.
    seed : optional
        Seed for the random number generator. If set to None, the seed is
        set to 0. If set to an integer, it is used as the seed directly.
        If set to a tuple, it is assumed to be a random state generated by
        np.random.get_state().

    Returns
    -------
    chi
        The mean-subtracted, variance-normalized \chi_e^2 GRF.

    Notes
    -----

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


############################################################
#
# 1D Line-of-Sight \chi_e^2 Field from P_k for 3D \chi_e^2 Field:
#
############################################################
def pk_chi_3d(k, amp, R, B=0.0):
    r"""Returns the power spectrum for a 3D \chi_e^2 GRF.
    
    Parameters
    ----------
    k
        Array of wavenumbers.
    amp
        Amplitude of the primordial power spectrum.
    R
        Parameter for the power spectrum.
    B : float, optional
        Parameter for the power spectrum.

    Returns
    -------
    pk
        The 3D power spectrum.

    Notes
    -----
    The power spectrum is given by P(k) = (2*pi^2 / k^3) * A * R^2 * k^2 * exp(-R^2 * k^2 + B).
    """

    pk = np.where(k!=0, ( (2*np.pi**2) / k**3 ) * amp * (R * k)**2 * np.exp( (- R**2 * k**2) + B), 0) # Power spectrum
    
    return pk

def grf_chi_3d_los1d(N, pk_amp, pk_R, pk_B=0.0, seed=None):
    r"""Generate a 1D, 'Line-Of-Sight' \chi_e^2 GRF from the power spectrum 
    generated for a 3D \chi_e^2 GRF.

    Parameters
    ----------
    N : int
        Size of the real space field.
    pk_amp
        Amplitude of the primordial power spectrum.
    pk_R
        Parameter for the power spectrum.
    pk_B : float, optional
        Parameter for the power spectrum.
    seed : optional
        Seed for the random number generator. If set to None, the seed is
        set to 0. If set to an integer, it is used as the seed directly.
        If set to a tuple, it is assumed to be a random state generated by
        np.random.get_state().

    Returns
    -------
    chi
        The variance-normalized \chi_e^2 GRF.

    Notes
    -----

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





############################################################
#
# Helper functions:
#
############################################################
# def power_array(Pk, k):
#     return np.where(k!=0, Pk(k), 0)
#     #return np.where(k==0, 0, Pk(k))

def gauss_var(size, seed=None):
    """Generates a complex Gaussian random variable in Fourier space with zero mean and unit variance.
    
    Parameters
    ----------
    size : int
        The size of the Gaussian random deviate array to be generated.
    seed : optional
        Seed for the random number generator. If set to None, the seed is
        set to 0. If set to an integer, it is used as the seed directly.
        If set to a tuple, it is assumed to be a random state generated by
        np.random.get_state().

    Returns
    -------
    a * (np.cos(e) + 1j * np.sin(e))
        The complex Gaussian random deviate array.

    Notes
    -----
    The random number generator is set to the seed if it is not None.
    """
    # If seed is provided as a tuple, use it to set the state of the random number generator
    if isinstance(seed, tuple):
        np.random.set_state(seed)
    # If seed is provided as an integer, use it to seed the random number generator
    elif isinstance(seed, int):
        np.random.seed(seed)
    # If seed is not provided, use the default seed value of 0
    elif seed is None:
        np.random.seed(0)
    else:
        raise ValueError("Invalid seed value: seed must be an integer or a tuple.")
    
    u = np.random.uniform(size=size)
    e = 2 * np.pi * np.random.uniform(size=size)
    # a = np.sqrt(-2*np.log(u))
    a = np.sqrt(-np.log(u))
    
    return a * (np.cos(e) + 1j * np.sin(e))

def dealiasx(f, kmaxknyq_ratio=(2/3)):
    """Dealias a real space field and return the de-aliased field in real space domain.
     
    Set the Fourier coefficients to zero if the wavenumber is greater than the maximum wavenumber (kmax), defined to be 2/3 of the Nyquist frequency.
    
    Parameters
    ----------
    f
        Input field in real space.
    kmaxknyq_ratio : float, optional
        The ratio of the maximum wavenumber to the Nyquist wavenumber.
    
    Returns
    -------
    ff
        The de-aliased field in real space.

    Notes
    -----
    The 2/3 ratio is commonly used in the literature.
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
    """Dealias a Fourier space field and return the de-aliased field in Fourier space domain.

    Set the Fourier coefficients to zero if the wavenumber is greater than the maximum wavenumber (kmax), defined to be 2/3 of the Nyquist frequency.

    Parameters
    ----------
    N
        The size of the real field array.
    fk
        Input field in Fourier space.
    k
        Array of wavenumbers.
    kmaxknyq_ratio : float, optional
        The ratio of the maximum wavenumber to the Nyquist wavenumber.

    Returns
    -------
    fk
        The de-aliased field in Fourier space.

    Notes
    -----
    The 2/3 ratio is commonly used in the literature.
    """

    knyq = N//2
    kmax = int( kmaxknyq_ratio * knyq )

    khigh = np.ones(k.size) * kmax
    fk = np.where(k!=0, np.where(np.less_equal(k, khigh), fk, 0), fk)

    return fk