# 
# Author: J. Haider
# 
"""
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compute_power_spectrum(field, bin_size=None):
    """
    Compute the power spectrum of a 1D field.

    Parameters
    ----------
    field : np.ndarray
        Input 1D field.
    bin_size : int, optional
        Size of the frequency bins for averaging the power spectrum.
        If None, no binning is applied. Default is None.

    Returns
    -------
    freqs : np.ndarray
        Frequencies corresponding to the power spectrum.
    power_spectrum : np.ndarray
        Power spectrum of the input field.
    """

    # Compute the Fourier transform of the field
    ft_field = np.fft.fft(field)

    # Calculate the power spectrum (absolute square)
    power_spectrum = np.abs(ft_field)**2

    # Normalize the power spectrum
    power_spectrum /= len(field)

    # Compute the frequencies corresponding to the power spectrum
    freqs = np.fft.fftfreq(len(field))

    # Apply binning if a bin size is specified
    if bin_size is not None:
        freqs = np.array([freqs[i:i + bin_size].mean() for i in range(0, len(freqs), bin_size)])
        power_spectrum = np.array([power_spectrum[i:i + bin_size].mean() for i in range(0, len(power_spectrum), bin_size)])

    return freqs, power_spectrum

# # Generate a sample 1D field (e.g., a sine wave)
# x = np.linspace(0, 4 * np.pi, 1000)
# field = np.sin(x)

# # Compute and plot the power spectrum with binning
# bin_size = 10
# freqs, power_spectrum = compute_power_spectrum(field, bin_size)
# plt.plot(freqs, power_spectrum)
# plt.xlabel("Frequency")
# plt.ylabel("Power")
# plt.show()



def calculate_cumulants_and_moments(zeta_smooth, cum, mom):
    """
    Calculate cumulants and moments for the given smoothed field.

    Parameters
    ----------
    zeta_smooth : np.ndarray
        Smoothed field for which cumulants and moments are to be calculated.
    cum : np.ndarray
        Pre-allocated array to store cumulants for the current bin.
    mom : np.ndarray
        Pre-allocated array to store moments for the current bin.
    """
    for i in range(4):
        cum[i] = stats.kstat(zeta_smooth, n=i+1)
        mom[i] = stats.moment(zeta_smooth, moment=i+1)

def calculate_field_statistics(fld, N, nbins, window_tophat_fn):
    """
    Calculate cumulants and moments for a given cosmological field in bins of wavenumber k.

    Parameters
    ----------
    fld : np.ndarray
        Input cosmological field (e.g., Gaussian random field or non-Gaussian field).
    N : int
        Size of the field.
    nbins : int
        Number of bins for k.
    window_tophat_fn : callable
        Function that applies a window filter to the field in k-space in the relevant bin-range.

    Returns
    -------
    cum : np.ndarray
        Cumulants for the given field in bins of k. Shape is (4, nbins).
    mom : np.ndarray
        Moments for the given field in bins of k. Shape is (4, nbins).
    k_bincentres : np.ndarray
        k bin centers corresponding to the calculated cumulants and moments.

    Examples
    --------
    >>> fld = fld_g  # or fld_ng, depending on which field you want to analyze
    >>> N = ...  # size of the field
    >>> nbins = 10  # number of bins
    >>> window_tophat_fn = flt.window_tophat
    >>> cum, mom, k_bincentres = calculate_field_statistics(fld, N, nbins, window_tophat_fn)
    """
    kc = np.geomspace(1, N//2, num=nbins+1)
    kc_size = kc.size
    log_kc = np.log2(kc)
    log_k_bincentres = (log_kc[:-1] + log_kc[1:]) / 2
    k_bincentres = 2**log_k_bincentres

    cum = np.zeros((4, nbins))
    mom = np.zeros((4, nbins))
    zeta_smooth = np.zeros((kc_size, N))

    for i in np.arange(0, nbins):
        zeta_smooth[i] = window_tophat_fn(fld, N, kc[i], kc[i+1])
        calculate_cumulants_and_moments(zeta_smooth[i], cum[:, i], mom[:, i])

    return cum, mom, k_bincentres












    
