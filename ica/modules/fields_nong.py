""" Generate Non-Gaussian Primodial Fields.
// Created by Jibran Haider & Tom Morrison. //

### Non-Gaussian Components of Fields:
<b>Set parameters and initialize the non-Gaussian components that will be added to the $\zeta$ GRFs.</b>

### Final Non-Gaussian Fields:
<b>Create the final, non-Gaussian $\zeta$ fields using the simulated GRF + the simulated non-G component.</b>

The 'ng_chisq' non-G component is uncorrelated to both 'zng1' and 'zng2'

**Contains the following functions**

"""


import numpy as np
import modules.fields_gauss as grf


############################################################
#
# Non-Gaussian Components of Fields:
# 
## <b>Set parameters and initialize the non-Gaussian components that will be added to the $\zeta$ GRFs.</b>
#
############################################################

### $\Chi_e^2$ non-Gaussianity
def nong_chisq(N, Achi=10**(-10), Rchi=0.04, Bchi=0.0, Fng=1.0, kmaxknyq_ratio=2/3, seedchi=None):
    """ Generate Chi_e^2 Non-Gaussianity.

    #
    # from nate:
    # Achi = 1.6*10**(-19)
    # Rchi = 0.64
    # Bchi = ?
    #

    """

    kmnr = kmaxknyq_ratio

    #
    # FINAL CHI_e^2 NON-G COMPONENT
    #
    grfchi = grf.grf_chi_1d(N, Achi, Rchi, Bchi, kmaxknyq_ratio=kmnr, seed=seedchi)
    ng_chisq = Fng * (grfchi)**2
    ng_chisq = grf.dealiasx(ng_chisq, kmaxknyq_ratio=kmnr)

    # s = np.std(ng_chisq)
    # ng_chisq = (ng_chisq / s)
    # ng_chisq = grf.dealiasx(ng_chisq, kmaxknyq_ratio=kmnr)
    # s = np.std(ng_chisq)
    # ng_chisq = (ng_chisq / s)

    return ng_chisq, grfchi


### Asymmetric $\sinh$ non-Gaussianity
## ???
def kth_root(x, n):
    """ ???.
    
    Used in nong_map_asymsinh below.
    """
    
    return np.where(x<0, -(-x)**(1/n), x**(1/n)).real

## Map input GRF to asymmetric sinh non-G
def nong_map_asymsinh(x, w, alpha):
    """ Map a GRF to an Asymmetric $\sinh$ Non-Gaussian component.
    
    """
    
    y = np.where(x<0, x, w * kth_root(np.sinh((x / w)**alpha), alpha))

    return y #- np.mean(y)

## Generate asymmetric sinh non-G component.
def nong_asymsinh(zg, nu=2, alpha=1.0, c=2, w=0.2):
    """ Generate Asymmetric Sinh Non-Gaussianity.

    Note that 'zg' is the field used to generate the NG component called 'ng_asymsinh'. So the 'zng_asymsinh' is correlated with 'zg'.
    In order to get uncorrelated 'nong_asymsinh', provide a GRF 'zg' different from the one used for the actual Primordial Field Realization.
    The final $\zeta$ field will be generated depending on the choice of 'correlation' (or lack thereof).

    """

    # Extract the standard deviation of the gaussian fields
    s = zg.std()

    #
    # FINAL ASYMMETRIC SINH NON-G COMPONENT
    #
    ng_asymsinh = nong_map_asymsinh(zg, nu*s, alpha) - zg

    return ng_asymsinh



############################################################
#
# Final Non-Gaussian Fields:
# 
## <b>Create the final, non-Gaussian $\zeta$ fields using the simulated GRF + the simulated non-G component.</b>
#
############################################################

### Add generic non-G to GRF to generate final Primordial field.
def nong_field(g, ng, FNL=1.0):
    """ Generate Primordial field with generic non-Gaussianity/FNL.
    
    """

    #
    # FINAL ZETA FIELD WITH NON-GAUSSIANITY
    #
    fieldng = g + FNL*ng
    
    return fieldng



### CHI_e^2 NON-GAUSSIAN FIELD
def nong_field_chisq(zg, Achi=10**(-10), Rchi=0.04, Bchi=0.0, Fng=1.0, kmaxknyq_ratio=2/3, seedchi=None):
    """ Generate FINAL Primordial Zeta field with uncorrelated Chi_e^2 Non-Gaussianity.
    
    #
    # from nate:
    # Achi = 1.6*10**(-19)
    # Rchi = 0.64
    # Bchi = ?
    #

    """

    N = zg.size
    kmnr = kmaxknyq_ratio

    #
    # FINAL CHI_e^2 NON-G COMPONENT
    #
    ng_chisq, grfchi = nong_chisq(N, Achi=Achi, Rchi=Rchi, Bchi=Bchi, Fng=Fng, kmaxknyq_ratio=kmnr, seedchi=seedchi)
    #
    # FINAL ZETA FIELD
    #
    zng_chisq = nong_field(zg, ng_chisq)

    return zng_chisq, ng_chisq, grfchi



### ASYMMETRIC SINH NON-GAUSSIAN FIELDS
## Correlated
def nong_field_asymsinh_corr(zg, nu=2, alpha=1.0, c=2, w=0.2):
    """ Generate FINAL Primordial Zeta field with Correlated Asymmetric Sinh Non-Gaussianity.

    Note that 'zg' is the field used to generate the NG component called 'ng_asymsinh'. 
    So the 'zng_asymsinh' is correlated with 'zg'.

    """

    zng_asymsinh = nong_asymsinh(zg, nu=nu, alpha=alpha, c=c, w=w)
    z_asymsinh = nong_field(zg, zng_asymsinh)

    return z_asymsinh, zng_asymsinh

## Uncorrelated
def nong_field_asymsinh_uncorr(zg1, zg2, nu=2, alpha=1.0, c=2, w=0.2):
    """ Generate FINAL Primordial Zeta field with Uncorrelated Asymmetric Sinh Non-Gaussianity.

    Note that 'zg1' is the field used to generate the NG component called 'ng_asymsinh'. 
    So the 'zng_asymsinh' is correlated with 'zg1' but uncorrelated with 'zg2', a different Gaussian random field.

    """

    zng_asymsinh = nong_asymsinh(zg1, nu=nu, alpha=alpha, c=c, w=w)
    z_asymsinh = nong_field(zg2, zng_asymsinh)

    return z_asymsinh, zng_asymsinh