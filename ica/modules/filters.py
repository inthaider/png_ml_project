"""
// Created by Jaafar.
Modified by Jibran Haider. //

"""


import numpy as np
from scipy.signal.windows import hann
from scipy.signal.windows import general_hamming as hamming

from modules.ica import ica_all

# import importlib as il
# import matplotlib.pyplot as plt 
# plt.rcParams.update({'font.size': 12}) 
# import matplotlib.gridspec as grd
# import pickle
# import nbodykit.lab as nbkt
# import modules.jaafar_fouriertransform as ft
# import modules.jaafar_peaks as px
# import modules.jaafar_correlationfunctions as cf
# import modules.gaussianfield as grf
# import modules.ica as ica


############################################################
#
# WINDOW FUNCTIONS
#
############################################################

# Top hat bands
def window_tophat(g, N, k_low, k_up):
    """
    
    """
    
    x = np.fft.rfft(g)
    # print(x[0:10])

    k = np.fft.rfftfreq(N) * N
    # print(k[0:10])

    # print(k[:N//2])

    k_low = np.ones(np.shape(x))*k_low
    k_up = np.ones(np.shape(x))*k_up

    print('x[0] before:', np.abs(x[0]))
    x = np.where(k!=0, (np.where(np.logical_and(np.less_equal(k, k_up), np.greater(k, k_low)), x, 0)), x)
    print('x[0] after:', np.abs(x[0]))
    
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

# Hann window
def window_hann(kbins):
    """Create Hann window filters in k-space.

    """

    kmin = kbins[0]
    kmax = kbins[-1]
    Nk = int(kmax - kmin)
    nkbins = int(kbins.size)
    skbin = Nk / (nkbins-1)

    hannfilts = np.zeros((nkbins, Nk))

    #
    #
    # DEAL WITH END BINS SEPARATELY
    #
    #
    for i in range(nkbins-1):
        kstart = kbins[i]
        kmid = kbins[i+1]

        if i == 0:
            kstop = kmid
            Nhann = kstop - kstart
            hannfilts[i, kstart:kstop] = hann(Nhann*2, False)[Nhann:]

        if i < nkbins-2:
            kstop = kbins[i+2]
            Nhann = kstop - kstart
            hannfilts[i+1, kstart:kstop] = hann(Nhann, False)
        else:
            kstop = kmid
            Nhann = kstop - kstart
            hannfilts[i+1, kstart:kstop] = hann(Nhann*2, False)[:Nhann]

    return hannfilts

def window_conv(gk, k, filt_win, kstart, kstop):
    """Apply window filter in k-space.
    
    """

    k_range = kstop - kstart

    kstart = np.ones(np.shape(gk))*kstart
    kstop = np.ones(np.shape(gk))*kstop

    # plt.plot(np.abs(gk))
    # plt.show()
    # plt.plot(filt_win)
    # plt.show()
    
    gk = np.where(np.logical_and(np.less(k, kstop), np.greater_equal(k, kstart)), gk*filt_win, 0)
    
    # plt.plot(np.abs(gk))
    # plt.show()
    
    return gk



############################################################
#
# FILTER STUFF
#
############################################################

def filter_hann(g, nkbins=5, k_min=None, k_max=None, dc_comp=False):
    """Filtering.

    """

    nkbins = int(nkbins)
    N = g.size
    gk = np.fft.rfft(g)
    dc = gk[0]
    k = np.fft.rfftfreq(N) * N
    Nk = int(k.size)

    # ############
    # plt.plot(k, np.abs(gk)/N)
    # plt.show()
    # ############

    if k_min == None and k_max == None:
        kmin = 0
        kmax = Nk
    elif k_min == None and k_max:
        kmin = 0
        kmax = Nk
    elif k_min and k_max == None:
        kmin = int(k_min)
        kmax = Nk
    else:
        kmin = int(k_min)
        kmax = int(k_max)

    k_trunc = k[kmin:kmax]
    # Nk_trunc = int(k_trunc.size)
    # skbin = (kmax - kmin) / (nkbins-1)
    kbins = np.round_(np.linspace(kmin, kmax, nkbins)).astype(int)
    print(kbins)
    hannfilts = window_hann(kbins)
    
    gk_trunc = gk[kmin:kmax]
    gk_filtered = np.zeros((nkbins, Nk), dtype=complex)
    gkt_filtered = np.zeros((nkbins, kmax-kmin), dtype=complex)
    g_filtered = np.zeros((nkbins, N))
    for i in range(nkbins):
        # ############
        # print(f"\nFiltering k-bin number:    {i} ...")
        # ############
        
        if i == 0:
            kstart = kbins[i]
            kstop = kbins[i+1]
            window = hannfilts[i, :]
            gk_filtered[i, kmin:kmax] = gkt_filtered[i, :] = window_conv(gk_trunc, k_trunc, window, kstart, kstop)
        elif i < nkbins-1:
            kstart = kbins[i-1]
            kstop = kbins[i+1]
            window = hannfilts[i, :]
            gk_filtered[i, kmin:kmax] = gkt_filtered[i, :] = window_conv(gk_trunc, k_trunc, window, kstart, kstop)
        else:
            kstart = kbins[i-1]
            kstop = kbins[i]
            window = hannfilts[i, :]
            gk_filtered[i, kmin:kmax] = gkt_filtered[i, :] = window_conv(gk_trunc, k_trunc, window, kstart, kstop)
        
        ############
        # plt.plot(np.abs(gk_filtered[i, :])/N)
        # plt.show()
        
        # print('start:', kstart, 'stop:', kstop)
        ############
        
        # tempk = window_conv(gk, k, window, kstart, kstop)
        # gk[kmin:kmax] = tempk
        if dc_comp:
            gk_filtered[i, 0] = dc
        temp = np.fft.irfft(gk_filtered[i, :])
        g_filtered[i, :] = temp
        
    return g_filtered, kbins, gkt_filtered, gk_trunc, hannfilts


##############################
# FILTERED ICA
##############################

def filterhann_ica(field_g, field_ng, 
                k_min=None, k_max=None, kmaxknyq_ratio=(2/3), nkbins=5, dc=False,
                    max_iter=1e4, tol=1e-5, fun='logcosh', whiten='unit-variance', algo='parallel', 
                        prewhiten = False, wbin_size = None):
    """

    """

    nkbins = int(nkbins)
    N = field_g.size
    k = np.fft.rfftfreq(N) * N
    Nk = int(k.size)

    kmnr = kmaxknyq_ratio
    knyq = N//2
    kmax_dealias = int( kmnr * knyq ) - 1

    #
    # Filtering parameters/vars
    #
    if k_min == None and k_max == None:
        kmin = 0
        kmax = kmax_dealias
    elif k_min == None and k_max:
        kmin = 0
        if k_max >= kmax_dealias:
            k_max = kmax_dealias
        kmax = int(k_max)
    elif k_min and k_max == None:
        kmin = int(k_min)
        kmax = kmax_dealias
    else:
        kmin = int(k_min)
        if k_max >= kmax_dealias:
            k_max = kmax_dealias
        kmax = int(k_max)

    #
    #
    # ICA parameters/vars
    #
    #
    ica_src = np.zeros((nkbins+1, 2, N))
    ica_src_og = np.zeros((nkbins+1, 2, N))
    src = np.zeros((nkbins+1, 2, N))
    max_amps = np.zeros((nkbins+1, 2, 3))
    zkt_filtered = np.zeros((nkbins, 2, kmax-kmin), dtype=complex)
    zkt = np.zeros((2, kmax-kmin), dtype=complex)

    print(f"Processing unfiltered field...")
    #
    # Run ICA
    #
    src[0, :], ica_src[0, :], max_amps[0, :], _, ica_src_og[0, :] = ica_all(field_g, field_ng, 
                                                                            max_iter=max_iter, tol=tol, fun=fun, whiten=whiten, algo=algo, 
                                                                            prewhiten = prewhiten, wbin_size = wbin_size)

    #
    # Filter
    #
    fzng, kbins, fzktng, zktng, _ = filter_hann(field_ng, nkbins=nkbins, k_min=kmin, k_max=kmax, dc_comp=dc)
    fzg, kbins, fzktg, zktg, hannf = filter_hann(field_g, nkbins=nkbins, k_min=kmin, k_max=kmax, dc_comp=dc)
    zkt_filtered[:, 0, :] = fzktng
    zkt_filtered[:, 1, :] = fzktg
    zkt[0, :] = zktng
    zkt[1, :] = zktg
    for i in range(nkbins):
        count = i+1
        print(f"Processing k-bin number:    {count} ...")

        zgf, zngf = fzg[i, :], fzng[i, :]
        
        # plt.plot(zngf)
        # plt.show()

        #
        # Run ICA
        #
        src[count, :], ica_src[count, :], max_amps[count, :], _, ica_src_og[count, :] = ica_all(zgf, zngf, 
                                            max_iter=max_iter, tol=tol, fun=fun, whiten=whiten, algo=algo, 
                                                prewhiten = prewhiten, wbin_size = wbin_size)
        src_max, ica_max = max_amps[0], max_amps[1]
    
    return src, ica_src, kbins, max_amps, zkt_filtered, zkt, hannf, ica_src_og




########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

def filterhat_gng(g_field, ng_field, size, k_low, k_high):
    """
    
    """

    g_field = window_tophat(g_field, size, k_low, k_high)
    ng_field = window_tophat(ng_field, size, k_low, k_high)

    return [g_field, ng_field]


def filterhat_ica(g, ng, 
                klow=None, khigh=None, nbins=10,
                    max_iter=1e4, tol=1e-5, fun='logcosh', whiten='unit-variance', algo='parallel', 
                        prewhiten = False, wbin_size = None):
    """Top hat filtering.

    """

    #
    #
    # Filtering parameters/vars
    #
    #
    size = g.size
    if not klow and not khigh:
        k_size = size//2 + 1
        k_low = 0
        k_high = k_size
    else:
        k_low = klow
        k_high = khigh
        
    kc = np.linspace(k_low, k_high, nbins+1)
    kc_size = kc.size

    #
    #
    # ICA parameters/vars
    #
    #
    ica_src = np.zeros((nbins+1, 2, size))
    src = np.zeros((nbins+1, 2, size))
    max_amps = np.zeros((nbins+1, 2, 3))

    #
    #
    # Run ICA
    #
    #
    src[0, :], ica_src[0, :], max_amps[0, :], _ = ica_all(g, ng, 
                                    max_iter=max_iter, tol=tol, fun=fun, whiten=whiten, algo=algo, 
                                        prewhiten = prewhiten, wbin_size = wbin_size)

    for i in range(nbins):
        count = i+1
        klow = kc[i]
        khigh = kc[i+1]

        print(f"Processing k-bin number:    {count} ...")

        #
        #
        # Filter
        #
        #
        filtered = filter(g, ng, size, int(klow), int(khigh))
        zgf, zngf = filtered[0], filtered[1]
        
        #
        #
        # Run ICA
        #
        #
        src[count, :], ica_src[count, :], max_amps[count, :], _ = ica_all(zgf, zngf, 
                                            max_iter=max_iter, tol=tol, fun=fun, whiten=whiten, algo=algo, 
                                                prewhiten = prewhiten, wbin_size = wbin_size)
        src_max, ica_max = max_amps[0], max_amps[1]
    
    return src, ica_src, kc, max_amps