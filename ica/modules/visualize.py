#
# Created by Jibran Haider.
#
r"""Visualization functions for ICA.

This module contains helper functions for ICA that can be used to plot source components and ICA-separated
signals and plot the ICA filter used for $kf$-ICA separation.

Routine Listings
----------------
plt_icaflt(src, ica_src, kc, max_amps, fontsize=8)
    Plot source components and ICA-separated signals
plt_filters(ica, kc, max_amps, fontsize=8)
    Plot the ICA filter used for $kf$-ICA separation.

See Also
--------
ica.modules.ica

Notes
-----

"""

import numpy as np
import matplotlib.pyplot as plt # plt.rcParams.update({'font.size': 12})

def plt_icaflt(src, ica_src, kc, max_amps, fontsize=8):
    '''Plot source components and ICA-separated signals
    
    Parameters
    ----------
    src
        Source components.
    ica_src
        ICA-separated signals
    kc
        wavenumber bin limits
    max_amps
        Maximum amplitudes of source components and ICA-separated signals.
    fontsize, optional
        Font size for plots.
    
    Returns
    -------
        None

    Notes
    -----
        1) Unfiltered source components are plotted in the top row.
        2) Filtered source components are plotted in the subsequent rows.
        3) The first column plots the source components.
        4) The second column plots the ICA-separated signals.
    '''
    N = src.shape[2]
    nbins = src.shape[0] - 1
    num_comps = src.shape[1]
    kc_size = kc.size

    #
    #
    # Plot
    #
    #
    src_max, ica_max = max_amps[0, 0, :], max_amps[0, 1, :]
    nrows = nbins + 1
    ncols = 2

    SMALL_SIZE = fontsize
    MEDIUM_SIZE = SMALL_SIZE+2
    BIGGER_SIZE = MEDIUM_SIZE+2

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE-2)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE-2)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    # plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(nrows, ncols, sharex='all', figsize=(6*ncols, 3*nrows), constrained_layout=True)

    offset = src_max[0]*1.8
    offset_ica = ica_max[0]*1.8

    ax00 = ax[0, 0]
    # Plotting source components
    ax[0, 0].set_title("(a) Source Components")
    for j in range(num_comps):
        if j == 0:
            label = "Non-Gaussian Component"
        else:
            label = "Gaussian Component"
        ax[0, 0].plot(src[0, j, :] + offset*j, label=label)
    ax[0, 0].set(ylabel=r"$\zeta$ amplitude (unfiltered)")
    ax[0, 0].legend(loc=1)

    ax01 = ax[0, 1]
    # Plotting ICA-separated signals
    ax[0, 1].set_title("(b) ICA-Separated Signals")
    # ax[0, 1].sharey(ax00)
    for j in range(num_comps):
        if j == 0:
            label = "Non-Gaussian Component"
        else:
            label = "Gaussian Component"
        ax[0, 1].plot(ica_src[0, j, :] + offset_ica*j, label=label) # Amplitudes are scaled arbitrarily because ICA doesn't recover amp
    # ax[0, 1].legend()

    ax[0, 0].text(0.5, 0.5, "UNFILTERED - FULL FIELD", 
                    fontsize='xx-large', transform=ax[0, 0].transAxes, 
                        ha='center', va='center', alpha=0.4)
    ax[0, 1].text(0.5, 0.5, "UNFILTERED - FULL FIELD", 
                    fontsize='xx-large', transform=ax[0, 1].transAxes, 
                        ha='center', va='center', alpha=0.4)
    ax[0, 0].legend(loc=1)

    for i in range(nbins):
        count = i+1
        if i == 0:
            klow = kc[i]
            khigh = kc[i+1]
        elif i < nbins-1:
            klow = kc[i-1]
            khigh = kc[i+1]
        else:
            klow = kc[i-1]
            khigh = kc[i]

        src_max, ica_max = max_amps[count, 0, :], max_amps[count, 1, :]
        offset_ = src_max[0]*1.6
        offset_ica_ = ica_max[0]*1.6
        klow = round(klow, 1); khigh = round(khigh, 1)

        # Plotting source components
        # ax[count, 0].sharey(ax00)
        for j in range(num_comps-1):
            if j == 0:
                label = "Non-Gaussian Component"
            else:
                label = "Gaussian Component"
            ax[count, 0].plot(src[count, j, :] + offset_*j, label=label)
        # ax[count, 0].set(ylabel=f'{i+1}) ' + "Zeta Amplitude with filter: " + r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh))
        ax[count, 0].set(ylabel=f'{i+1}) ' + r"$\zeta$ amplitude")
        
        # ax[count, 1].sharey(ax00)
        # Plotting ICA-separated signals
        for j in range(num_comps-1):
            if j == 0:
                label = "Non-Gaussian Component"
            else:
                label = "Gaussian Component"
            ax[count, 1].plot(ica_src[count, j, :] + offset_ica_*j, label=label) # Amplitudes are scaled arbitrarily because ICA doesn't recover amp

        ax[count, 0].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
                                fontsize='x-large', transform=ax[count, 0].transAxes, 
                                    ha='center', va='center', alpha=0.4)
        ax[count, 1].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
                                fontsize='x-large', transform=ax[count, 1].transAxes, 
                                    ha='center', va='center', alpha=0.4)

    ax_count = kc_size
    ax[ax_count, 0].set(xlabel=r'$x$')
    ax[ax_count, 1].set(xlabel=r'$x$')

    fig.suptitle(rf'Filtered $\it{{FastICA}}$-separation with $k: [{{{kc[0]}}}, {{{kc[-1]}}}]$.' + f'\nField size: {N}.', fontsize=BIGGER_SIZE)

    note="Note: The Gaussian components are manually offset up from 0 for the purpose of clarity."
    fig.text(0.5, -0.01, note, wrap=True, horizontalalignment='center', fontsize=MEDIUM_SIZE)

    plt.savefig(f'/Users/JawanHaider/Desktop/research/research_projects/pnong_ml/ica/figures/icafiltered/chie2/chie2_kfica_s{N}_{int(kc[0])}to{int(kc[-1])}k{nbins}.png', facecolor='white', bbox_inches='tight')
    plt.show()

    return


def plt_filters(N, kc, fzkt, zkt, hannf, fontsize=8):
    '''Plot the filters used in the filtering process.
    
    Parameters
    ----------
    N
        field size
    kc
        The wavenumber bins limits.
    fzkt
        Filtered Fourier transform of the source components.
    zkt
        Fourier transform of the source components.
    hannf
        Hann window
    fontsize, optional
        Font size.
    
    Returns
    -------
        None. Plots the filters used in the filtering process.

    Notes
    -----
        The filters are plotted in the frequency domain.
    '''
    fzkt = np.abs(fzkt) / N
    zkt = np.abs(zkt) / N

    Nk = fzkt.shape[2]
    nkbins = fzkt.shape[0]
    ncomps = fzkt.shape[1]

    #
    #
    # Plot
    #
    #
    nrows = nkbins + 1
    ncols = 2

    SMALL_SIZE = fontsize
    MEDIUM_SIZE = SMALL_SIZE+2
    BIGGER_SIZE = MEDIUM_SIZE+2

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE-2)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE-2)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    # plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(nrows, 1, sharex='all', figsize=(6, 3*nrows), constrained_layout=True)

    ax0 = ax[0]
    for i in range(nkbins):
        # if i == 0:
        #     klow = kc[i]
        #     khigh = kc[i+1]
        # elif i < nkbins-1:
        #     klow = kc[i-1]
        #     khigh = kc[i+1]
        # else:
        #     klow = kc[i-1]
        #     khigh = kc[i]
            
        if i < nkbins-1:
            axx = ax[0].twinx()
            color = 'tab:red'
            axx.plot(np.nonzero(hannf[i, :])[0], hannf[i, hannf[i, :]!=0], color=color, alpha=0.4)
            axx.tick_params(axis='y', labelcolor=color)
            axx.tick_params(
                axis='y',           # changes apply to the y-axis
                which='both',       # both major and minor ticks are affected
                right=False,        # ticks along the right edge are off
                labelright=False)   # labels along the right edge are off
        else:
            axx = ax[0].twinx()
            color = 'tab:red'
            label = "Hann window"
            axx.plot(hannf[i, :], label=label, color=color, alpha=0.4)
            axx.tick_params(axis='y', labelcolor=color)
            axx.set_ylabel('Window Amplitude', color=color)
            axx.legend(loc=1)
    
    ax[0].set_title("Unfiltered k-frequencies with Hann")
    label = "k-frequencies"
    ax[0].plot(zkt[0, :], label=label)
    ax[0].set(ylabel=r"$k$ amplitude (unfiltered)")
    ax[0].legend(loc=2)

    ax[0].text(0.5, 0.5, "UNFILTERED K-FREQUENCIES", 
                    fontsize='xx-large', transform=ax[0].transAxes, 
                        ha='center', va='center', alpha=0.4)

    ax[1].set_title("Filtered k-frequencies with Hann")
    for i in range(nkbins):
        count = i+1
        if i == 0:
            klow = kc[i]
            khigh = kc[i+1]
        elif i < nkbins-1:
            klow = kc[i-1]
            khigh = kc[i+1]
        else:
            klow = kc[i-1]
            khigh = kc[i]

        klow = round(klow, 0); khigh = round(khigh, 0)


        axx = ax[count].twinx()
        color = 'tab:red'
        axx.set_ylabel('Window Amplitude', color=color)
        label = "Hann window"
        axx.plot(hannf[i, :], label=label, color=color)
        axx.tick_params(axis='y', labelcolor=color)
        axx.legend(loc=1)

        # ax[count, 0].sharey(ax00)
        label = "k-frequencies"
        ax[count].plot(fzkt[i, 0, :], label=label)
        # ax[count].set(ylabel=f'{count}) ' + "k amplitude with filter: " + r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh))
        ax[count].set(ylabel=f'{count}) ' + r"$k$ amplitude")
        ax[count].legend(loc=2)
        

        ax[count].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
                                fontsize='x-large', transform=ax[count].transAxes, 
                                    ha='center', va='center', alpha=0.4)

    ax_count = nkbins
    ax[ax_count].set(xlabel=r'$k$')

    fig.suptitle(rf'Hann window-filtering in Fourier domain: $k=[{{{kc[0]}}}, {{{kc[-1]}}}]$' + f'\nField size: {N}.', fontsize=BIGGER_SIZE)

    import os
    print(os.getcwd())  
    plt.savefig(f'/Users/JawanHaider/Desktop/research/research_projects/pnong_ml/ica/figures/icafiltered/chie2/chie2_hann_s{N}_{int(kc[0])}to{int(kc[-1])}k{nkbins}.png', facecolor='white', bbox_inches='tight')
    # note="Note: The Gaussian components are manually offset up from 0 for the purpose of clarity."
    # fig.text(0.5, -0.01, note, wrap=True, horizontalalignment='center', fontsize=8)
    plt.show()

    return