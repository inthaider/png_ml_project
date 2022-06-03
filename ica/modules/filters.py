import numpy as np
from scipy.signal import windows, get_window
from scipy.signal.windows import general_hamming as hamming
import modules.fouriertransform as ft


# Top hat bands
def window(g, N, k_low, k_up):
    """
    
    """
    
    x = np.fft.rfft(g)
    # print(x[0:10])

    k = np.fft.rfftfreq(N) * N
    # print(k[0:10])

    # print(k[:N//2])

    k_low = np.ones(np.shape(x))*k_low
    k_up = np.ones(np.shape(x))*k_up

    x = np.where(np.logical_and(np.less_equal(k, k_up), np.greater(k, k_low)), x, 0)
    x_inv = np.fft.irfft(x)

    return x_inv

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

    x = np.where(np.logical_and(np.less_equal(k, k_high), np.greater_equal(k, k_low)), x*hamm, 0)
    x_inv = np.fft.irfft(x)
    
    return x_inv, hamm

def filter(g_field, ng_field, size, k_low, k_high):
    """
    
    """

    g_field = window(g_field, size, k_low, k_high)
    ng_field = window(ng_field, size, k_low, k_high)

    return [g_field, ng_field]





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

