"""
// Created by Jibran Haider. //

"""


import numpy as np
import matplotlib.pyplot as plt # plt.rcParams.update({'font.size': 12})


def resid(a, b):
    """Compute both scalar & vector residuals between $a$ & $b$.

    Args:
        a (): Vector.
        b (): Vector.
        
    Returns:
        rr (float?): Scalar residual of $a$ and $b$ (see below).
        rv (): Vector residual of $a$ and $b$ (see below).
    

    Notes:
        1) Normalize $b$ by mean-subtraction & std-division.

        2) Rescale $b$ by $a$'s std.
        
        3) The vector residual ($rv$) is calculated as:
            rv = ( ( b.a / a.a ) * a ) / |a|
        
        4) The scalar residual ($rr$) is calculated as:
            rr = 1 - | ( b.a / |a| ) / |a| |
        
        Note that:
            |x| = (x.x)^{1/2}
        where $x$ is a vector.

    TODO:
        FILL IN WHY I CHOSE TO COMPUTE $RV$ & $RR$ THIS WAY!
    """
    a_std = np.std(a)
    a_mean = np.mean(a)
    b_std = np.std(b)
    b_mean = np.mean(b)
    
    b = ((b - b_mean) / b_std) 
    b = b * a_std

    bdota = np.dot(b, a)
    adota = np.dot(a, a)
    amag = np.sqrt(adota)

    rv = ( (bdota / adota) * a ) / amag
    # rv = 1 - (bdota / adota)
    r = np.linalg.norm(rv, 2)

    anorm = np.linalg.norm(a, 2)
    bnorm = np.linalg.norm(b, 2)

    rr = 1 - np.abs(((bdota / amag) / amag))
    # rr = 1 - r

    # ab = np.abs(1 - anorm / bnorm)

    return rr, rv


def plt_icaflt(src, ica_src, kc, max_amps, fontsize=7):
    """

    """

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
    plt.rcParams.update({'font.size': fontsize})
    nrows = nbins + 1
    ncols = 2

    fig, ax = plt.subplots(nrows, ncols, sharex='all', figsize=(9*ncols, 4*nrows), constrained_layout=True)

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
    ax[0, 0].set(ylabel="Zeta amplitude without filtering.")
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
        ax[count, 0].set(ylabel=f'{i+1}) ' + "Zeta Amplitude with filter: " + r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh))
        
        # ax[count, 1].sharey(ax00)
        # Plotting ICA-separated signals
        for j in range(num_comps-1):
            if j == 0:
                label = "Non-Gaussian Component"
            else:
                label = "Gaussian Component"
            ax[count, 1].plot(ica_src[count, j, :] + offset_ica_*j, label=label) # Amplitudes are scaled arbitrarily because ICA doesn't recover amp

        ax[count, 0].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
                                fontsize='xx-large', transform=ax[count, 0].transAxes, 
                                    ha='center', va='center', alpha=0.4)
        ax[count, 1].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
                                fontsize='xx-large', transform=ax[count, 1].transAxes, 
                                    ha='center', va='center', alpha=0.4)

    ax_count = kc_size-1
    ax[ax_count, 0].set(xlabel=r'$x$')
    ax[ax_count, 1].set(xlabel=r'$x$')

    fig.suptitle(rf'Filtered $\it{{FastICA}}$-separation with $k: [{{{kc[0]}}}, {{{kc[-1]}}}]$.' + f'\nField size: {N}.', fontsize=16)

    note="Note: The Gaussian components are manually offset up from 0 for the purpose of clarity."
    fig.text(0.5, -0.01, note, wrap=True, horizontalalignment='center', fontsize=8)
    
    plt.savefig(f'/fs/lustre/cita/haider/projects/pnong_ml/ica/figures/icafiltered/chie2/chie2_kfica_s{N}_{int(kc[0])}to{int(kc[-1])}k{nbins}.png', facecolor='white', bbox_inches='tight')
    plt.show()

    return


def plt_filters(N, kc, fzkt, zkt, hannf, fontsize=7):
    """

    """

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
    plt.rcParams.update({'font.size': fontsize})
    nrows = nkbins + 1
    ncols = 2

    fig, ax = plt.subplots(nrows, 1, sharex='all', figsize=(8, 4*nrows), constrained_layout=True)

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
    ax[0].set(ylabel="k-amplitude (unfiltered)")
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
        ax[count].set(ylabel=f'{count}) ' + "k amplitude with filter: " + r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh))
        ax[count].legend(loc=2)
        

        ax[count].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
                                fontsize='xx-large', transform=ax[count].transAxes, 
                                    ha='center', va='center', alpha=0.4)

    ax_count = nkbins
    ax[ax_count].set(xlabel=r'$k$')

    fig.suptitle(rf'Hann window-filtering in Fourier-domain with $k: [{{{kc[0]}}}, {{{kc[-1]}}}]$.' + f'\nField size: {N}.', fontsize=16)

    plt.savefig(f'/fs/lustre/cita/haider/projects/pnong_ml/ica/figures/icafiltered/chie2/chie2_hann_s{N}_{int(kc[0])}to{int(kc[-1])}k{nkbins}.png', facecolor='white', bbox_inches='tight')
    # note="Note: The Gaussian components are manually offset up from 0 for the purpose of clarity."
    # fig.text(0.5, -0.01, note, wrap=True, horizontalalignment='center', fontsize=8)
    plt.show()

    return