import numpy as np

def fftmodes(N, BoxSize=None):
    #return np.fft.fftshift(np.fft.fftfreq(N, N**2))
    if BoxSize is None:
        return np.fft.fftshift(np.fft.fftfreq(N))
    return np.fft.fftshift(np.fft.fftfreq(N))*(N/BoxSize)

def fft(x, BoxSize=1.0, modes=False, grid=False):
    
    N = x.shape[0]
    
    if not grid and not modes :
        return (BoxSize)**-3*np.fft.fftshift(np.fft.fftn(x))
    
    k = fftmodes(N)
    
    if not grid and modes : 
        return (BoxSize)**-3*np.fft.fftshift(np.fft.fftn(x)), k
    
    kx, ky, kz = np.meshgrid(k, k, k)
    g = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    
    if grid and not modes:
        return (BoxSize)**(-3)*np.fft.fftshift(np.fft.fftn(x)), g
    
    if grid and modes:
        return (BoxSize)**(-3)*np.fft.fftshift(np.fft.fftn(x)), k, g



def ifft(x, BoxSize=1.0):
    #N = x.shape[0]
    return np.fft.ifftn(np.fft.ifftshift(x))*(BoxSize)**3

