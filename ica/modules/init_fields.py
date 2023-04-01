## !/usr/bin/env python3
"""Imports Peak-Patch fields as NumPy arrays.

@Authors:   Jibran Haider & Nathan Carlson

Examples
--------
I'm currently setup to run in terminal with a command::

    $ python3 

Or you can run me in a .ipynb notebook with::

    %run 

Or you can import me as a module::

    import 

Attributes
----------
delta : 

delta_g : 

delta_ng : 

zeta : 

zeta_g : 

zeta_ng : 

X : 

Y : 

Z : 

Notes
-----
Example Peak-Patch realization: 
    peak-patch-runs/n1024bigR/z2/fnl1e6/
    New runs: 
            /mnt/scratch-lustre/njcarlson/peak-patch-runs/ng7_test_oct6
                Then there’s 3D fields in the “fields" subdirectory, and there’s 2D healpix maps in the “maps" subdirectory, you can play with those with python using healpy.
            /mnt/scratch-lustre/njcarlson/peak-patch-runs/s4k_n236_nb20_nt10_ng6_nside2048/fnl5e3/
            
TODO
----
Write necessary code to be able to turn Delta fields processing on or off.

"""

#----Import modules----#

from ast import Not
from importlib.resources import path
from pathlib import Path # For path manipulations and module loading

import numpy as np
from numpy.random import randint as nprandint

# Import local module 'get_params'
# to import relevant field parameters.
import modules.sim_params as sim_params

#----Initialize variables----#

global l_mpc
global l_array
global l_buff
global l_trim
global fields_path

d_filename = 'Fvec_17Mpc_n1024_nb64_nt1'
dg_filename = 'rhog_17Mpc_n1024_nb64_nt1'
z_filename = 'zetang_17Mpc_n1024_nb64_nt1'
zg_filename = 'zetag_17Mpc_n1024_nb64_nt1'

d_filename = 'Fvec_128Mpc_n128_nb40_nt1'
dg_filename = 'rhog_128Mpc_n128_nb40_nt1'
z_filename = 'zetang_128Mpc_n128_nb40_nt1'
zg_filename = 'zetag_128Mpc_n128_nb40_nt1'

d_filename = 'Fvec_4000Mpc_n236_nb20_nt10'
dg_filename = 'rhog_4000Mpc_n236_nb20_nt10'
z_filename = 'zetang_4000Mpc_n236_nb20_nt10'
zg_filename = 'zetag_4000Mpc_n236_nb20_nt10'
filenames = [d_filename, dg_filename, z_filename, zg_filename]

#----Delta fields----#

def import_params(path_realization: str | Path = None, lmpc = None, larray = None, lbuff = None):
    """
    
    """
    
    global l_mpc
    global l_array
    global l_buff
    global l_trim
    global fields_path
    
    l_mpc = lmpc
    l_array = larray
    l_buff = lbuff
    l_trim = l_array - l_buff*2
    fields_path = Path(Path(path_realization)/"fields")

    print(l_mpc, l_array, l_buff, l_trim)
    print(not (l_mpc))
    print(not (l_mpc or l_array or l_buff or l_trim))
    if not (l_mpc or l_array or l_buff or l_trim):
        l_mpc, l_array, l_buff, l_trim, fields_path = sim_params.main(fields_path)

    return

def get_delta(file_name: str = None):
    """Import total Delta field (G + nonG).

    """
    
    if not file_name:
        file_name = filenames[0]

    # Total non-Gaussian delta field
    delta_file = fields_path/file_name
    in_delta   = open(delta_file, 'rb')
    # Read in delta, reshape it into an nxnxn, and then trim off the buffers
    delta = np.fromfile(in_delta,dtype=np.float32,count=-1)
    delta = np.reshape(delta, (l_array,l_array,l_array), order='F')
    if l_buff != 0:
        delta = delta[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]

    return delta

def get_delta_g(file_name: str = None):
    """Import Gaussian component of Delta field.

    """
        
    if not file_name:
        file_name = filenames[1]

    # Gaussian delta field
    delta_g_file = fields_path/file_name
    in_delta_g   = open(delta_g_file, 'rb')

    # Read in delta_g, reshape it into an nxnxn, and then trim off buffers
    delta_g = np.fromfile(in_delta_g,dtype=np.float32,count=-1) 
    delta_g = np.reshape(delta_g, (l_array,l_array,l_array), order='F')
    if l_buff != 0:
        delta_g = delta_g[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]

    return delta_g

def get_delta_ng(delta, delta_g):
    """Import nonG component of Delta (delta - delta_g = delta_ng).

    """
        
    # if not d_file_name:
    #     file_name = filenames[0]
    # if not dg_file_name:
    #     file_name = filenames[1]

    # nonG component of Delta
    delta_ng = delta - delta_g
    
    return delta_ng

def get_delta_all(d_file_name: str = None, dg_file_name: str = None):
    """Import Delta fields
    
    """
    
    print('\nProcessing Delta fields/components...\n')
    delta = get_delta(d_file_name)
    delta_g = get_delta_g(dg_file_name)
    delta_ng = get_delta_ng(delta, delta_g)

    return delta, delta_g, delta_ng

#----Zeta fields----#

def get_zeta(file_name: str = None):
    """Import total Zeta field (G + nonG).

    """
    print('\nGetting zeta total...\n')

    if not file_name:
        file_name = filenames[2]

    # non-Gaussian zeta field
    print(fields_path/file_name)
    zeta_file = fields_path/file_name
    in_zeta   = open(zeta_file, 'rb')
    # Read in zeta, reshape it into an nxnxn, and then trim off the buffers
    zeta = np.fromfile(in_zeta,dtype=np.float32,count=-1)
    zeta = np.reshape(zeta, (l_array,l_array,l_array), order='F')
    if l_buff != 0:
        print('Removing buffers...')
        zeta = zeta[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]
    
    print('Zeta total shape:', zeta.shape)
    return zeta

def get_zeta_g(file_name: str = None):
    """Import Gaussian component of Zeta field.

    """
    print('\nGetting zeta Gauss...\n')

    if not file_name:
        file_name = filenames[3]

    # Gaussian zeta field
    print(fields_path/file_name)
    zeta_g_file = fields_path/file_name
    in_zeta_g   = open(zeta_g_file, 'rb')
    # Read in zeta_g, reshape it into an nxnxn, and then trim off buffers
    zeta_g = np.fromfile(in_zeta_g,dtype=np.float32,count=-1)
    zeta_g = np.reshape(zeta_g, (l_array,l_array,l_array), order='F')
    if l_buff != 0:
        print('Removing buffers...')
        zeta_g = zeta_g[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]

    print('Zeta Gauss shape:', zeta_g.shape)
    return zeta_g

def get_zeta_ng(zeta, zeta_g):
    """Import nonG component of Zeta (zeta - zeta_g = zeta_ng).

    """
    print('\nGetting zeta nonG...\n')

    # nonG component of Zeta
    zeta_ng = zeta - zeta_g

    print('Zeta nonG shape:', zeta_ng.shape)
    return zeta_ng

def get_zeta_all(z_file_name: str = None, zg_file_name: str = None):
    """Import Zeta fields
    
    """
    
    print('\nProcessing Zeta fields/components...\n')
    print(z_file_name)
    print(zg_file_name)
    zeta = get_zeta(z_file_name)
    zeta_g = get_zeta_g(zg_file_name)
    zeta_ng = get_zeta_ng(zeta, zeta_g)

    print('\nDone processing Zeta fields/components!\n')
    return zeta, zeta_g, zeta_ng

#--------------------------------------------------#

def get_meshgrid(field_side_mpc, array_side: int):
    """Initialize a meshgrid.

    """

    # Defines X,Y,Z as meshgrid
    edges = np.linspace( -field_side_mpc/2 , field_side_mpc/2 , array_side+1 )
    X,Y,Z = np.meshgrid(edges,edges,edges,indexing='ij')

    return X,Y,Z

#--------------------------------------------------#

def main(path_realization: str | Path = None, lengths=None, isDelta=False):
    """Main function.

    TODO:
        Write necessary code to be able to turn Delta fields processing on or off.
    """

    import_params(path_realization, lengths[0], lengths[1], lengths[2])

    """Import Delta fields"""
    if isDelta:
        delta, delta_g, delta_ng = get_delta_all(d_filename, dg_filename)
    # else:
    #     delta, delta_g, delta_ng = (None, None, None)
    #     print("\n NOTE: Not extracting Delta fields (Deltas returned will be valued 'None'.)... \n")

    """Import Zeta fields"""
    zeta, zeta_g, zeta_ng = get_zeta_all(z_filename, zg_filename)

    """
    You now have zeta_g, delta_g, and delta, which are three n-by-n-by-n NumPy arrays representing a gaussian zeta field, a gaussian density field (specifically rho bar times delta, that we talked about today) and a non-gaussian delta field. 
    """

    """Check entries from extracted fields"""
    print('\nChecking random entries from each of the fields...\n')

    # Check random entries from each of the extracted fields
    # Random coordinates
    x, y, z = nprandint(0, l_trim, 3)

    if isDelta:
        print('\nDelta Gauss-comp ({}, {}, {}):      '.format(x, y, z), delta_g[x,y,z])
        print('Delta nonG-comp ({}, {}, {}):        '.format(x, y, z), delta_ng[x,y,z])
        print('Delta total ({}, {}, {}):            '.format(x, y, z), delta[x,y,z])

    print('\nZeta Gauss-comp ({}, {}, {}):       '.format(x, y, z), zeta_g[x,y,z])
    print('Zeta nonG-comp ({}, {}, {}):         '.format(x, y, z), zeta_ng[x,y,z])
    print('Zeta total ({}, {}, {}):             '.format(x, y, z), zeta[x,y,z])

    return [delta, delta_g, delta_ng, zeta, zeta_g, zeta_ng] if isDelta else [zeta, zeta_g, zeta_ng]

#--------------------------------------------------#












#--------------------------------------------------#

if __name__=="__main__":

    path_realization = Path('peak-patch-runs/n1024bigR/z2/fnl1e6')
    path_realization = Path('/mnt/scratch-lustre/njcarlson/peak-patch-runs/ng7_test_oct6/')
    path_realization = Path('/mnt/scratch-lustre/njcarlson/peak-patch-runs/s4k_n236_nb20_nt10_ng6_nside2048/fnl5e3/')

    lmpc = 4000; larray = 2000; lbuff = 0;
    lengths = [lmpc, larray, lbuff]
    
    import_params(path_realization, lengths[0], lengths[1], lengths[2])
    """Import Delta fields"""
    delta, delta_g, delta_ng = get_delta_all(d_filename, dg_filename)

    """Import Zeta fields"""
    zeta, zeta_g, zeta_ng = get_zeta_all(z_filename, zg_filename)

    """
    You now have zeta_g, delta_g, and delta, which are three n-by-n-by-n NumPy arrays representing a gaussian zeta field, a gaussian density field (specifically rho bar times delta, that we talked about today) and a non-gaussian delta field. 
    """

    """Check entries from extracted fields"""
    print('\nChecking random entries from each of the fields...\n')
    
    # Check random entries from each of the extracted fields
    # Random coordinates
    x, y, z = nprandint(0, l_trim, 3)

    print('\nDelta Gauss-comp ({}, {}, {}):      '.format(x, y, z), delta_g[x,y,z])
    print('Delta nonG-comp ({}, {}, {}):        '.format(x, y, z), delta_ng[x,y,z])
    print('Delta total ({}, {}, {}):            '.format(x, y, z), delta[x,y,z])

    print('\nZeta Gauss-comp ({}, {}, {}):       '.format(x, y, z), zeta_g[x,y,z])
    print('Zeta nonG-comp ({}, {}, {}):         '.format(x, y, z), zeta_ng[x,y,z])
    print('Zeta total ({}, {}, {}):             '.format(x, y, z), zeta[x,y,z])

#--------------------------------------------------#