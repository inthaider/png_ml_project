## !/usr/bin/env python3
"""Slices 1D strips from a given realization of 3D Peak-Patch fields.

@Authors:   Jibran Haider & Nathan Carlson

Examples
--------
TESTI'm currently setup to run in terminal with a command::

    $ python3 slicer_1d.py

Or you can run me in a .ipynb notebook with::

    %run slicer_1d.py

Or you can import me as a module::

    import slicer_1d

Attributes
----------

Notes
-----
Example Peak-Patch realization: 
    peak-patch-runs/n1024bigR/z2/fnl1e6/

"""

#----Import modules----#

from pathlib import Path # For path manipulations and module loading

import numpy as np
import numpy.random as nprandom
from numpy.random import seed as npseed
from numpy.random import rand as nprand
from numpy.random import randint as nprandint

# import get_params
import init_fields
import sim_params

#--------------------------------------------------#

class Slicer1D:
    """1D slicer object for a given initial fields realization.
    
    Attributes:
        self.idx_seed (int) : Reseed the MT19937 BitGenerator.
        self.path_pkp_realization (str | Path) : Path to initial fields realization
        self.fields_3d (list) : Full 3D initial fields.
        self.side_length (int) : Side length of the box after trimming off the buffers.

    """    

    def __init__(self, idx_seed=None, 
            path_pkp_realization: str | Path=None, 
            fields_3d : list=None, 
            is_rand_axes : bool=True):
        """Initialise 1D slicer object for a given initial fields realization.

        Input:
            path_pkp_realization (str | Path) : Path to initial fields realization
            idx_seed : Seed for the MT19937 BitGenerator.
            fields_3d : Full 3D initial fields
            is_rand_axes : 
            

        """

        self.idx_seed = idx_seed
        self.path_pkp_realization = path_pkp_realization
        self.is_rand_axes = is_rand_axes
        self.indices : tuple = ()
        self.fields_1d : list = []

        if fields_3d == None:
            self.fields_3d = init_fields.main(self.path_pkp_realization)
            self.side_length = init_fields.l_trim
        else:
            self.fields_3d = fields_3d
            self.side_length = self.fields_3d[0].shape[0]

        return

    def set_seed(self, seed):
        """
        
        """

        self.idx_seed = seed

        return

    def slice_1d(self, is_rand_axes : bool = True):    
        """Extract 1D strips from the 3D fields.

        Args:
            field_3d (list): 3D fields from which to extract a 1D strip.
            indices (tuple): List of indices to slice 3D field-array with.
        
        Returns:
            fields_1d (): 1D strips corresponding to list of 3D fields in 'field_3d'.
        """

        indices = self.idx_rand_slice(is_rand_axes)
        
        indices = self.indices
        fields_3d = self.fields_3d

        self.fields_1d = [i[indices] for i in fields_3d]

        return self.fields_1d

    def idx_rand_slice(self, is_rand_axes : bool = True):
        """Generate random indices to take 1D slice from 3D field.

        Args:
            idx_seed (int): Reseed the MT19937 BitGenerator.
            is_rand_axes (bool): Whether to randomize axes or not. Defaults to True.
            
        
        Returns:
            coords (tuple): List of indices to slice 3D field-array with.
        """
        
        seed = self.idx_seed
        side_length = self.side_length

        # Initialize seed for randomization
        # if not given, a random seed will be initialized
        if isinstance(seed, int):
            npseed(seed)
        else:
            nprandom.set_state(seed)

        # TWO random coordinates, and a SINGLE, 1D, full slice of the field
        x, y, z = tuple(nprandint(0, side_length, 2))+(range(0, side_length),)
        coords = [x, y, z]

        if is_rand_axes:
            # Shuffle the axes to truly pick out a random 1D slice
            nprandom.shuffle(coords)

        self.indices = tuple(coords)

        return self.indices

#--------------------------------------------------#

def main(seed = None):
    """Main function.
    
    """

    # fields_3d = init_fields.main()
    # side_length = init_fields.l_trim

    # idx = idx_rand_slice(side_length, seed)
    # dg_1d, dng_1d, d_1d, zg_1d, zng_1d, z_1d = slice_1d(fields_3d, idx)

    # return [dg_1d, dng_1d, d_1d, zg_1d, zng_1d, z_1d]
    pass

#--------------------------------------------------#

if __name__=="__main__":

    path_realization = 'peak-patch-runs/n1024bigR/z2/fnl1e6/'
    seed = 12412

    slicer = Slicer1D(seed, path_realization)
    print(slicer.idx_seed, slicer.side_length, slicer.indices[0])
    fields = slicer.slice_1d()
    d_1d, dg_1d, dng_1d, z_1d, zg_1d, zng_1d = fields

    print(d_1d.shapes)

    

#--------------------------------------------------#