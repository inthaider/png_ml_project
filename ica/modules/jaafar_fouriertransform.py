"""
// Created by Jaafar.
Modified by Jibran Haider & Tom Morrison. //


**Contains the following functions**

fftmodes(N, BoxSize=None):
fft(x, BoxSize=1.0, modes=False, grid=False, oneD=False):
ifft(x, BoxSize=1.0):

"""


import numpy as np


def fftmodes(N, BoxSize=None):
    """
    RETURNS:

    np.fft.fftshift(np.fft.fftfreq(N))  IF  no BoxSize given

    OR
    
    np.fft.fftshift(np.fft.fftfreq(N)) * (N/BoxSize)  IF  BoxSize is given 
    (this just corrects the frequencies to match the intended, physical size of the box instead of N)


    np.fft() is numpy's discrete fast fourier transform.
    np.fft.fftfreq() returns the Discrete Fourier Transform sample frequencies 
    [where you can provide the sample spacing, i.e. BoxSize/N in this case.]
    np.fft.fftshift() shifts the zero-frequency component to the center of the spectrum.

    """
    #return np.fft.fftshift(np.fft.fftfreq(N, N**2))

    if BoxSize is None:
        return np.fft.fftshift(np.fft.fftfreq(N))
    return np.fft.fftshift(np.fft.fftfreq(N, (BoxSize/N) ))

def fft(x, BoxSize=1.0, modes=False, grid=False, oneD=False):
    
    N = x.shape[0]
    
    if not grid and not modes :
        return (BoxSize)**-3*np.fft.fftshift(np.fft.fftn(x))
    
    k = fftmodes(N)
    
    if not grid and modes : 
        return (BoxSize)**-3*np.fft.fftshift(np.fft.fftn(x)), k
    
    if oneD :
        g = np.sqrt(k ** 2)
    else :
        kx, ky, kz = np.meshgrid(k, k, k)
        g = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    
    if grid and not modes:
        return (BoxSize)**(-3)*np.fft.fftshift(np.fft.fftn(x)), g
    
    if grid and modes:
        return (BoxSize)**(-3)*np.fft.fftshift(np.fft.fftn(x)), k, g



def ifft(x, BoxSize=1.0):
    #N = x.shape[0]
    return np.fft.ifftn(np.fft.ifftshift(x))*(BoxSize)**3

