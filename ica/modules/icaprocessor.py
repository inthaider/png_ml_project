#
# Created by Jibran Haider.
#
"""This module contains the ICA class for performing ICA on a given signal mixture.

Classes
-------
ICA
    Class for performing Independent Component Analysis (ICA) on a given signal mixture.

Notes
-----
"""
import numpy as np
from sklearn.decomposition import FastICA

class ICAProcessor:
    """Class for performing Independent Component Analysis (ICA) on a given signal mixture.

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations to run ICA.
    tol : float
        Tolerance for ICA convergence.
    fun : str
        The functional form of the G function used in the ICA algorithm.
    whiten : str
        The whitening method to use.
    algo : str
        The algorithm to use.
    prewhiten : bool
        Whether to prewhiten the data before running ICA.
    wbin_size : int
        The size of the bins to use for prewhitening.
    src_max : ndarray
        The maximum values of the source components.
    ica_max : ndarray
        The maximum values of the ICA components.

    Methods
    -------
    ica_setup(source_noise, source_nonG)
        Set up signal mixture for ICA.
    fastica_run(mix, num_comps)
        Initialize FastICA with given params.
    ica_all(field_g, field_ng)
        Preprocess signals, run ICA, and perform postprocessing on the given fields.
    match_rescale_ica(src_comps, ica_comps)
        Match and rescale ICA components to the original source components.
    create_comps_dict(comps, comp_order=["PNG", "GRF"])
        Create a dictionary of field components from an array of components.
    find_max(src_comps, ica_comps)
        Find maximum amplitude values (positive or negative) for both source and ICA-separated data.
    """
    def __init__(self, max_iter=1e4, tol=1e-5, fun='logcosh', whiten='unit-variance', algo='parallel', prewhiten=False, wbin_size=None):
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.whiten = whiten
        self.algo = algo
        self.prewhiten = prewhiten
        self.wbin_size = wbin_size
        self.src_max = None
        self.ica_max = None

    def ica_setup(self, source_grf, source_png):
        """Set up signal mixture for ICA.

        Parameters
        ----------
        source_noise : array
            GRF component of the signal mixture.
        source_nonG : array
            PNG component of the signal mixture.

        Returns
        -------
        mix_signal : array
            Mixed signal.
        source_comps : array
            Source components.
        num_comps : int
            Number of source components.
        """

        source_comps = np.vstack([source_png, source_grf])
        num_comps = source_comps.shape[0]
        num_samples = num_comps

        mix_matrix = (1.0+np.random.random((num_samples, num_comps)))/2.0
        mix_signal = np.dot(mix_matrix, source_comps) # mixed signals

        return mix_signal, source_comps, num_comps

    def fastica_run(self, mix, num_comps):
        """Initialize FastICA with given params.

        Parameters
        ----------
        mix : np.ndarray, shape (n, m)
            nxm numpy array containing the mixed/observed signals.
        num_comps : int
            Number of components to extract.
        max_iter : int, optional
            Maximum number of iterations to run FastICA. The default is 1e4.
        tol : float, optional
            Tolerance for convergence. The default is 1e-5.
        fun : str, optional
            Cost-function to use for ICA. The default is 'logcosh'.
        whiten : str, optional
            Whitening method to use. The default is 'unit-variance'.
        algo : str, optional
            Algorithm to use for ICA. The default is 'parallel'.

        Returns
        -------
        sources.T : np.ndarray, shape (m, n)
            mxn numpy array containing the extracted source components.

        Notes
        -----
        Logcosh is negentropy.
        """
        
        # , white='unit-variance'
        transformer = FastICA(n_components=num_comps, algorithm=self.algo, whiten=self.whiten, max_iter=self.max_iter, tol=self.tol, fun=self.fun)

        # run FastICA on observed (mixed) signals
        sources = transformer.fit_transform(mix.T)

        # print(transformer.components_.shape)
        
        return sources.T

    def match_rescale_ica(self, src_comps, ica_comps):
        """Match and rescale ICA components to the original source components.

        Parameters
        ----------
        src_comps : dict
            Dictionary containing the source components labeled "GRF" and "PNG".
        ica_comps : np.ndarray, shape (2, n)
            2xn numpy array containing the ICA extracted components.

        Returns
        -------
        matched_comps : dict
            Dictionary of properly labeled, rescaled, and sign-inverted ICA components.
        """

        def calc_residual(a, b):
            """
            Calculate the residual between vectors a and b.

            This function rescales b to match the variance of a and then calculates
            the residual, which is insensitive to sign differences.
            """
            b = (b / np.std(b)) * np.std(a)
            # b_norm = b_rescaled / np.linalg.norm(b_rescaled)
            # projection = np.dot(a, b_norm)
            # 1 - np.abs(projection / np.linalg.norm(a))

            # Calculate dot products and scalar projection
            bdota = np.dot(b, a)
            adota = np.dot(a, a)
            mag_a = np.sqrt(adota)      # magnitude of x: |x| = (x.x)^{1/2}
            scalar_proj_ab = bdota/mag_a

            resid = 1 - np.abs(scalar_proj_ab) / mag_a        # rs = 1 - |(b.a / |a|) / |a||; absolute value to ensure positive scalar projection

            return np.abs(resid)       # return absolute value of scalar residual since we want the relative scalar residual to be positive
        
        def rescale_invert(a, b):
            """Rescale, mean-calibrate, and invert vector b to match vector a.

            This function rescales b to match the variance and mean of a and inverts its sign
            if the projection of b onto a is negative.
            Use np.dot(a, b) / np.dot(b, b) to match the variance and signs.
            """
            b = b - np.mean(b)
            # b = (b / np.std(b)) * np.std(a)
            # if np.dot(a, b) < 0:
            #     b = -b
            scale_factor = np.dot(a, b) / np.dot(b, b)
            b = scale_factor * b
            b = b + np.mean(a)
            return b
            # return scale_factor * b

        matched_comps = {}
        for label, src_comp in src_comps.items():
            min_residual = np.inf       # initialize minimum residual to infinity
            best_match = None
            best_match_idx = -1
            
            print("label: ", label)
            print("ica_comps shape: ", ica_comps.shape)
            
            for idx, ica_comp in enumerate(ica_comps):
                
                print("idx: ", idx)
                print("src_comp shape: ", src_comp.shape)
                print("ica_comp shape: ", ica_comp.shape)

                residual = calc_residual(src_comp, ica_comp)
                if residual < min_residual:
                    min_residual = residual
                    best_match = ica_comp
                    best_match_idx = idx
                    
            rescaled_inverted_comp = rescale_invert(src_comp, best_match)
            matched_comps[label] = rescaled_inverted_comp
            ica_comps = np.delete(ica_comps, best_match_idx, axis=0)

        return matched_comps

    def ica_all(self, field_g, field_ng):
        """Preprocess signals, run ICA, and perform postprocessing on the given fields.

        Parameters
        ----------
        field_g : np.ndarray, shape (n, m)
            n x m numpy array containing the field data for the Gaussian source.
        field_ng : np.ndarray, shape (n, m)
            n x m numpy array containing the field data for the non-Gaussian source.
        max_iter : int, optional
            Maximum number of iterations to run ICA. The default is 1e4.
        tol : float, optional
            Tolerance for ICA convergence. The default is 1e-5.
        fun : str, optional
            The functional form of the G function used in the ICA algorithm. The default is 'logcosh'.
        whiten : str, optional
            The whitening method to use. The default is 'unit-variance'.
        algo : str, optional
            The algorithm to use. The default is 'parallel'.
        prewhiten : bool, optional
            Whether to prewhiten the data before running ICA. The default is False.
        wbin_size : int, optional
            The size of the bins to use for prewhitening. The default is None.

        Returns
        -------
        src : np.ndarray, shape (2, n)
            2xn numpy array containing the source components.
        ica_src : np.ndarray, shape (2, n)
            2xn numpy array containing the ICA components.
        src_max : np.ndarray, shape (2,)
            2x1 numpy array containing the maximum values of the source components.
        ica_max : np.ndarray, shape (2,)
            2x1 numpy array containing the maximum values of the ICA components.
        mix_signal : np.ndarray, shape (2, n)
            2xn numpy array containing the mixed signals.
        ica_src_og : np.ndarray, shape (2, n)
            2xn numpy array containing the ICA components before postprocessing.    
        """
        
        mix_signal_pre, src, num_comps = self.ica_setup(field_g, field_ng)
        if self.prewhiten:
            # mix_signal = ica_prewhiten(mix_signal_pre, wbin_size)
            pass
        else:
            mix_signal = mix_signal_pre

        ica_src_og = self.fastica_run(mix_signal, num_comps)
        
        # Convert source components to a dictionary with labels for PNG and GRF
        # The default order in create_comps_dict is PNG, GRF
        src_comps_dict = self.create_comps_dict(src)
        print("src_comps_dict dimensions: ", src_comps_dict["PNG"].shape, src_comps_dict["GRF"].shape)
        print("ica_src_og shape: ", ica_src_og.shape)
        print("src_comps_dict: ", src_comps_dict)
        ica_comps_dict = self.match_rescale_ica(src_comps_dict, ica_src_og)
        # ica_comps_dict = self.create_comps_dict(ica_src)

        # Calculate the maximum values of the source and ICA components
        max_dict = self.find_max(src_comps_dict, ica_comps_dict)
        src_max = max_dict["Source"]
        ica_max = max_dict["ICA"]

        # Convert ICA components dictionary to a numpy array
        ica_src = np.vstack([ica_comps_dict["PNG"], ica_comps_dict["GRF"]])
        return src, ica_src, np.array([src_max, ica_max]), np.array([mix_signal_pre, mix_signal]), ica_src_og

    @staticmethod
    def create_comps_dict(comps, comp_order=["PNG", "GRF"]):
        """Create a dictionary of field components from an array of components.

        Parameters
        ----------
        comps : np.ndarray, shape (2, n)
            2xn numpy array containing the field components.
        comp_order : list, optional
            List of strings containing the order of the components. The default is ["PNG", "GRF"].
            The first element should be the label for the first component, and so on. For example,
            if the first component is the PNG field component, then the first element
            of the list should be "PNG". 

        Returns
        -------
        comps_dict : dict
            Dictionary containing the field components labeled "GRF" and "PNG".

        """
        comps_dict = {}
        # enumerate(comps.T) returns a tuple of the index and the component, i.e. (0, comp0), (1, comp1), etc.
        for idx, comp in enumerate(comps):
            comps_dict[comp_order[idx]] = comp
        return comps_dict
    
    @staticmethod
    def find_max(src_comps, ica_comps):
        """Find maximum amplitude values (positive or negative) for both source and ICA-separated data.
        
        This function takes in two dictionaries, source and ICA, with comps labelled
        'PNG' and 'GRF' comps and finds 3 maximum values for each of the dictionaries: 
        the max of the whole dictionary, the max of just the 'PNG' comp, and 
        the max of just the 'GRF' comp.
        
        The maximum values are found by taking the absolute value, taking into account 
        only the amplitude of the peaks, not the sign.
        
        It outputs the max values as one dictionary with the keys "Source" and "ICA", and
        the values as an ndarray of the following 3 maximum values for each of "Source" and "ICA": 
            [max of whole dictionary, max of PNG comp, max of GRF comp].
        
        Parameters
        ----------
        src_comps : dict
            Dictionary containing the source components labeled "GRF" and "PNG".
        ica_comps : dict
            Dictionary containing the ICA-separated components labeled "GRF" and "PNG".

        Returns
        -------
        max_dict : dict
            Dictionary containing the 3 maximum values for each of the source and ICA datasets.
        """
        # Find the maximum values for the source and ICA-separated data
        max_dict = {}
        for key, comps in zip(["Source", "ICA"], [src_comps, ica_comps]):
            max_dict[key] = np.zeros(3)
            # Extract the max of the whole dictionary
            max_dict[key][0] = np.max(np.abs(np.concatenate([comps["PNG"], comps["GRF"]])))
            # Extract the max of the PNG and GRF comps
            max_dict[key][1] = np.max(np.abs(comps["PNG"]))
            max_dict[key][2] = np.max(np.abs(comps["GRF"]))
        
        return max_dict