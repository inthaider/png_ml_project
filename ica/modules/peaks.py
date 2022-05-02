import numpy as np

# Correlated peaks functions

def kth_root(x, n):
    return np.where(x<0, -(-x)**(1/n), x**(1/n)).real

def map_sinh(x, w, alpha):
    y = w * kth_root(np.sinh((x / w)**alpha), alpha)
    return y - np.mean(y)
    #return np.where(x)

def map_bypart(x, xp, a):
    y = np.where(x<xp, x, a*x)
    return y - np.mean(y)

def map_asymm_sinh(x, w, alpha):
    y = np.where(x<0, x, w * kth_root(np.sinh((x / w)**alpha), alpha))
    return y #- np.mean(y)

def map_bump(x, z1, z2):
    y = np.where(x<z1, x, np.where(x>z2, x, z1))
    return y - np.mean(y)

def map_smooth_bump(x, c, a):
    y = np.abs(x-c)*np.tanh((x-c)/a) + c
    return y - np.mean(y)

def sq_ng(x, f_nl, s2):
    return x + f_nl*(x**2-s2)
    

# Uncorrelated peaks functions

def peak_profile(x, y, z, a, s):
    #return a * (2 * np.pi * s**2)**-1.5 * np.exp(-.5 * (x**2 + y**2 + z**2) / s**2)
    return a * np.exp(-.5 * (x**2 + y**2 + z**2) / s ** 2)

def add_peaks(X, L, size):
    
    '''L is a list of [[x, y, z], zp, rp], where xp is the location of the 
    peak/dip, zp its height and rp its width
    '''
    
    l = len(L)
    P = X.copy()
    l_x = np.arange(0, size)
    x_p, y_p, z_p = np.meshgrid(l_x, l_x, l_x)
    
    
    for i in range(l):
        P += peak_profile(x_p-L[i][0][0], y_p-L[i][0][1], z_p-L[i][0][2],
                          L[i][1], L[i][2])
    return P

def draw_peaks(nop, size, zp, rp, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randint(low=0, high=size, size=(nop,3))
    
    if isinstance(zp, (list, tuple, np.ndarray)) and isinstance(rp, (list, tuple, np.ndarray)):
        return [[[x[i][0], x[i][1], x[i][2]], zp[i], rp[i]] for i in range(nop)]
    
    if isinstance(zp, (list, tuple, np.ndarray)):
        return [[[x[i][0], x[i][1], x[i][2]], zp[i], rp] for i in range(nop)]
    
    if isinstance(rp, (list, tuple, np.ndarray)):
        return [[[x[i][0], x[i][1], x[i][2]], zp, rp[i]] for i in range(nop)]
    
    return [[[x[i][0], x[i][1], x[i][2]], zp, rp] for i in range(nop)]

