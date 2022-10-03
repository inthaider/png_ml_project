# #
# #
# # Filtering parameters/vars
# #
# #
# nbins = 10
# k_size = size//2 + 1
# k_low = 0
# kl_global = k_low
# k_high = k_size
# kc = np.linspace(0, k_high, nbins+1)
# # kc = np.array([0, 20, 40, 80])
# kc_size = kc.size

# #
# #
# # ICA parameters/vars
# #
# #
# max_iter = int(9e13)
# tol = 1e-12
# ica_src = np.zeros((kc_size+1, 2, size))

# #
# #
# # Run ICA
# #
# #
# mix_signal, src, num_comps = ica_setup(zg, zng)
# # mix_signal = ica_preprocess(mix_signal, 100)
# ica_src_og = ica_run(mix_signal, num_comps, max_iter, tol)
# ica_src[0, :], src_max, ica_max = ica_prepres(src, ica_src_og)


# #
# #
# # Plot
# #
# #
# plt.rcParams.update({'font.size': 7})
# nrows = nbins + 1
# ncols = 2

# fig, ax = plt.subplots(nrows, ncols, sharex='all', figsize=(6*ncols, 3*nrows), constrained_layout=True)

# offset = src_max[0]*1.8
# offset_ica = ica_max[0]*1.8

# ax00 = ax[0, 0]
# # Plotting source components
# ax[0, 0].set_title("(a) Source Components")
# for j in range(num_comps):
#     if j == 0:
#         label = "Non-Gaussian Component"
#     else:
#         label = "Gaussian Component"
#     ax[0, 0].plot(src[j, :] + offset*j, label=label)
# ax[0, 0].set(ylabel="Zeta amplitude without filtering.")
# ax[0, 0].legend(loc=1)

# ax01 = ax[0, 1]
# # Plotting ICA-separated signals
# ax[0, 1].set_title("(b) ICA-Separated Signals")
# ax[0, 1].sharey(ax00)
# for j in range(num_comps):
#     if j == 0:
#         label = "Non-Gaussian Component"
#     else:
#         label = "Gaussian Component"
#     ax[0, 1].plot(ica_src[0, j, :] + offset_ica*j, label=label) # Amplitudes are scaled arbitrarily because ICA doesn't recover amp
# # ax[0, 1].legend()

# ax[0, 0].text(0.5, 0.5, "UNFILTERED - FULL FIELD", 
#                 fontsize='xx-large', transform=ax[0, 0].transAxes, 
#                     ha='center', va='center', alpha=0.4)
# ax[0, 1].text(0.5, 0.5, "UNFILTERED - FULL FIELD", 
#                 fontsize='xx-large', transform=ax[0, 1].transAxes, 
#                     ha='center', va='center', alpha=0.4)
# ax[0, 0].legend(loc=1)

# for i in range(kc_size-1):
#     count = i+1
#     klow = kc[i]
#     khigh = kc[i+1]

#     print(f"\nProcessing k-bin number:    {count} ...")

#     #
#     #
#     # Filter
#     #
#     #
#     filtered = filter(zg, zng, size, int(klow), int(khigh))
#     zgf, zngf = filtered[0], filtered[1]
    
#     #
#     #
#     # Run ICA
#     #
#     #
#     mix_signal, src, num_comps = ica_setup(zgf, zngf)
#     # mix_signal = ica_preprocess(mix_signal, 100)
#     ica_src_og = ica_run(mix_signal, num_comps, max_iter, tol)
#     ica_src[count, :], src_max, ica_max = ica_prepres(src, ica_src_og)

#     offset_ = src_max[0]*1.8
#     offset_ica_ = ica_max[0]*1.8
#     klow = round(klow, 1); khigh = round(khigh, 1)

#     # Plotting source components
#     ax[count, 0].sharey(ax00)
#     for j in range(num_comps):
#         if j == 0:
#             label = "Non-Gaussian Component"
#         else:
#             label = "Gaussian Component"
#         ax[count, 0].plot(src[j, :] + offset*j, label=label)
#     ax[count, 0].set(ylabel=f'{i+1}) ' + "Zeta Amplitude with filter: " + r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh))
#     # ax[count, 0].legend()
    
#     ax[count, 1].sharey(ax00)
#     # Plotting ICA-separated signals
#     for j in range(num_comps):
#         if j == 0:
#             label = "Non-Gaussian Component"
#         else:
#             label = "Gaussian Component"
#         ax[count, 1].plot(ica_src[count, j, :] + offset_ica*j, label=label) # Amplitudes are scaled arbitrarily because ICA doesn't recover amp
#     # ax[count, 1].legend()

#     a = src[0, :]
#     b = ica_src[count, 0, :]
#     bdota = np.dot(b, a)
#     adota = np.dot(a, a)
#     rv = b - (bdota / adota) * a
#     r = np.linalg.norm(rv, 2)
#     anorm = np.linalg.norm(a, 2)
#     rr = r/anorm


#     print("residual (input vs ouput nonG): ", rr)

#     ax[count, 0].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
#                             fontsize='xx-large', transform=ax[count, 0].transAxes, 
#                                 ha='center', va='center', alpha=0.4)
#     ax[count, 1].text(0.5, 0.5, r"$k=[{{{kl}}}, {{{kh}}}]$".format(kl=klow, kh=khigh), 
#                             fontsize='xx-large', transform=ax[count, 1].transAxes, 
#                                 ha='center', va='center', alpha=0.4)

# ax_count = kc_size-1
# ax[ax_count, 0].set(xlabel=r'$x$')
# ax[ax_count, 1].set(xlabel=r'$x$')

# fig.suptitle(rf'Filtered $\it{{FastICA}}$-separation with $k: [{{{k_low}}}, {{{k_high}}}]$.' + f'\nField size: {size}.', fontsize=16)

# note="Note: The Gaussian components are manually offset up from 0 for the purpose of clarity."
# fig.text(0.5, -0.01, note, wrap=True, horizontalalignment='center', fontsize=8)
# plt.show()

# plt.savefig(f'/fs/lustre/cita/haider/projects/pnong_ml/ica/plots/icafiltered_s{size}_{int(kl_global)}to{int(khigh)}k{nbins}.png')

