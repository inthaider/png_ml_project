# 
# Created by Jibran Haider & Tom Morrison.
# 
"""This module contains functions for generating various forms of Non-Gaussian Primodial components and fields.

The non-Gaussian components are added to the Gaussian Random Fields (GRFs) to create the final, non-Gaussian Primordial Fields.

Routine Listings
----------------
png_chisq
    Generate Chi_e^2 Non-Gaussianity (only the FNG component, not the whole field) and return its correlated GRF too.
png_asymsinh
    Generate Asymmetric Sinh Non-Gaussianity (only the FNG component, not the full NG field).
png_map_asymsinh
    Map a GRF to an Asymmetric $\sinh$ Non-Gaussian component.
kth_root
    Return the kth root of x.

png_field_chisq
    Generate the full Chi_e^2 Non-Gaussian Field.
png_field_asymsinh_corr
    Generate the full Asymmetric Sinh Non-Gaussian Field with correlated components.
png_field_asymsinh_uncorr
    Generate the full Asymmetric Sinh Non-Gaussian Field with uncorrelated components.

png_field
    Helper function to generate the full Non-Gaussian Field for a given PNG component.

Notes
-----
The 'ng_chisq' non-G component is uncorrelated to both 'zng1' and 'zng2'
"""

import numpy as np
import ica.modules.fields_gauss as grf
# from . import fields_gauss as grf

############################################################
#
# Non-Gaussian Components of Fields:
# 
## <b>Set parameters and initialize the non-Gaussian components that will be added to the $\zeta$ GRFs.</b>
#
############################################################
### $\Chi_e^2$ non-Gaussianity
def png_chisq(N, Achi=10**(-10), Rchi=0.04, Bchi=0.0, Fng=1.0, kmaxknyq_ratio=2/3, seedchi=None):
    """Generate Chi_e^2 Non-Gaussianity (only the FNG component, not the whole field) and return its correlated GRF too.

    Parameters
    ----------
    N : int
        Number of grid points.
    Achi : float
        Amplitude of the non-Gaussian component.
    Rchi : float
        Correlation length of the non-Gaussian component.
    Bchi : float
        Bias of the non-Gaussian component.
    Fng : float
        ???
    kmaxknyq_ratio : float
        Fraction of the Nyquist frequency to use as the maximum wavenumber.
    seedchi : int
        Seed for the random number generator.

    Returns
    -------
    ng_chisq : array
        Non-Gaussian component of the field.
    grf_chisq : array
        Gaussian random field.

    Notes
    -----
    from nate:
        Achi = 1.6*10**(-19)
        Rchi = 0.64
        Bchi = ?
        
    This makes the ng_chisq and grf_chisq fields correlated to each other.
    """

    kmnr = kmaxknyq_ratio

    #
    # FINAL CHI_e^2 NON-G COMPONENT
    #
    grf_chisq = grf.grf_chi_1d(N, Achi, Rchi, Bchi, kmaxknyq_ratio=kmnr, seed=seedchi)
    ng_chisq = Fng * (grf_chisq)**2
    ng_chisq = grf.dealiasx(ng_chisq, kmaxknyq_ratio=kmnr)

    # s = np.std(ng_chisq)
    # ng_chisq = (ng_chisq / s)
    # ng_chisq = grf.dealiasx(ng_chisq, kmaxknyq_ratio=kmnr)
    # s = np.std(ng_chisq)
    # ng_chisq = (ng_chisq / s)

    return ng_chisq, grf_chisq


## Generate asymmetric sinh non-G component.
def png_asymsinh(zg, nu=2, alpha=1.0, c=2, w=0.2):
    """Generate Asymmetric Sinh Non-Gaussianity (only the FNG component, not the full NG field).

    Parameters
    ----------
    zg : array
        Gaussian random field.
    nu : float
        ???
    alpha : float
        ???
    c : float
        ???
    w : float
        ???

    Returns
    -------
    ng_asymsinh : array
        Non-Gaussian component of the field.

    Notes
    -----
    Note that 'zg' is the field used to generate the NG component called 'ng_asymsinh'. So the 'zng_asymsinh' is correlated with 'zg'.
    In order to get uncorrelated 'png_asymsinh', provide a GRF 'zg' different from the one used for the actual Primordial Field Realization.
    The final $\zeta$ field will be generated depending on the choice of 'correlation' (or lack thereof).
    """

    # Extract the standard deviation of the gaussian fields
    s = zg.std()

    #
    # FINAL ASYMMETRIC SINH NON-G COMPONENT
    #
    ng_asymsinh = png_map_asymsinh(zg, nu*s, alpha) - zg

    return ng_asymsinh

## Map input GRF to asymmetric sinh non-G
def png_map_asymsinh(x, w, alpha):
    r"""Map a GRF to an Asymmetric $\sinh$ Non-Gaussian component.
    
    Parameters
    ----------
    x
        The input GRF
    w
        The width of the distribution
    alpha
        The power of the $\sinh$ function
    
    Returns
    -------
    y
        The value of y, which is the value of x if x is less than 0, and the value of w *
    kth_root(np.sinh((x / w)**alpha), alpha) if x is greater than 0.
    """
    
    y = np.where(x<0, x, w * kth_root(np.sinh((x / w)**alpha), alpha))

    return y #- np.mean(y)

### Asymmetric $\sinh$ non-Gaussianity
## ???
def kth_root(x, n):
    """Return the kth root of x.

    Parameters
    ----------
    x : float or array
        The number/array of numbers to take the root of.
    n : float
        The power of the root.

    Returns
    -------
    array
        The kth root of x.
    
    Notes
    -----
    Used in png_map_asymsinh above.
    """
    
    return np.where(x<0, -(-x)**(1/n), x**(1/n)).real



############################################################
#
# Final Non-Gaussian Fields:
# 
## <b>Create the final, non-Gaussian $\zeta$ fields using the simulated GRF + the simulated non-G component.</b>
#
############################################################
### CHI_e^2 NON-GAUSSIAN FIELD
def png_field_chisq(zg, Achi=10**(-10), Rchi=0.04, Bchi=0.0, Fng=1.0, kmaxknyq_ratio=2/3, seedchi=None):
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
    ng_chisq, grfchi = png_chisq(N, Achi=Achi, Rchi=Rchi, Bchi=Bchi, Fng=Fng, kmaxknyq_ratio=kmnr, seedchi=seedchi)
    #
    # FINAL ZETA FIELD
    #
    zng_chisq = png_field(zg, ng_chisq)

    return zng_chisq, ng_chisq, grfchi


### ASYMMETRIC SINH NON-GAUSSIAN FIELDS
## Correlated
def png_field_asymsinh_corr(zg, nu=2, alpha=1.0, c=2, w=0.2):
    """ Generate FINAL Primordial Zeta field with Correlated Asymmetric Sinh Non-Gaussianity.

    Note that 'zg' is the field used to generate the NG component called 'ng_asymsinh'. 
    So the 'zng_asymsinh' is correlated with 'zg'.

    """

    zng_asymsinh = png_asymsinh(zg, nu=nu, alpha=alpha, c=c, w=w)
    z_asymsinh = png_field(zg, zng_asymsinh)

    return z_asymsinh, zng_asymsinh

## Uncorrelated
def png_field_asymsinh_uncorr(zg1, zg2, nu=2, alpha=1.0, c=2, w=0.2):
    """ Generate FINAL Primordial Zeta field with Uncorrelated Asymmetric Sinh Non-Gaussianity.

    Note that 'zg1' is the field used to generate the NG component called 'ng_asymsinh'. 
    So the 'zng_asymsinh' is correlated with 'zg1' but uncorrelated with 'zg2', a different Gaussian random field.

    """

    zng_asymsinh = png_asymsinh(zg1, nu=nu, alpha=alpha, c=c, w=w)
    z_asymsinh = png_field(zg2, zng_asymsinh)

    return z_asymsinh, zng_asymsinh


### Add generic non-G to GRF to generate final Primordial field.
def png_field(g, ng, FNL=1.0):
    """ Generate Primordial field with generic non-Gaussianity/FNL.
    
    """

    #
    # FINAL ZETA FIELD WITH NON-GAUSSIANITY
    #
    fieldng = g + FNL*ng
    
    return fieldng