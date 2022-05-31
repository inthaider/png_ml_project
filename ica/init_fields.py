## !/usr/bin/env python3
"""Imports Peak-Patch fields as NumPy arrays.

@Authors:   Jibran Haider & Nathan Carlson

Examples
--------
TESTI'm currently setup to run in terminal with a command::

    $ python3 get_fields.py

Or you can run me in a .ipynb notebook with::

    %run get_fields.py

Or you can import me as a module::

    import get_fields

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

"""

#----Import modules----#

from pathlib import Path # For path manipulations and module loading

import numpy as np
from numpy.random import randint as nprandint

# Import local module 'get_params'
# to import relevant field parameters.
import sim_params

#----Initialize variables----#

l_mpc=0.0; l_array=0; l_buff=0; l_trim=0; fields_path : Path = Path()

d_filename = 'Fvec_fNL_17Mpc_n1024_nb64_nt1'
dg_filename = 'Fvec_17Mpc_n1024_nb64_nt1'
z_filename = 'zetang_17Mpc_n1024_nb64_nt1'
zg_filename = 'zetag_17Mpc_n1024_nb64_nt1'
filenames = [d_filename, dg_filename, z_filename, zg_filename]

#----Delta fields----#

def import_params(path_realization: str | Path = None):
    """
    
    """
    
    global l_mpc
    global l_array
    global l_buff
    global l_trim
    global fields_path

    l_mpc, l_array, l_buff, l_trim, fields_path = sim_params.main(path_realization)

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
    delta_g = delta_g[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]

    return delta_g

def get_delta_ng(d_file_name: str = None, dg_file_name: str = None):
    """Import nonG component of Delta (delta - delta_g = delta_ng).

    """
        
    # if not d_file_name:
    #     file_name = filenames[0]
    # if not dg_file_name:
    #     file_name = filenames[1]

    # nonG component of Delta
    delta_ng = get_delta(d_file_name) - get_delta_g(dg_file_name)
    
    return delta_ng

def get_delta_all(d_file_name: str = None, dg_file_name: str = None):
    """Import Delta fields
    
    """
    
    print('\nProcessing Delta fields/components...\n')
    delta = get_delta(d_file_name)
    delta_g = get_delta_g(dg_file_name)
    delta_ng = get_delta_ng(d_file_name, dg_file_name)

    return delta, delta_g, delta_ng

#----Zeta fields----#

def get_zeta(file_name: str = None):
    """Import total Zeta field (G + nonG).

    """

    if not file_name:
        file_name = filenames[2]

    # non-Gaussian zeta field
    zeta_file = fields_path/file_name
    in_zeta   = open(zeta_file, 'rb')
    # Read in zeta, reshape it into an nxnxn, and then trim off the buffers
    zeta = np.fromfile(in_zeta,dtype=np.float32,count=-1)
    zeta = np.reshape(zeta, (l_array,l_array,l_array), order='F')
    zeta = zeta[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]

    return zeta

def get_zeta_g(file_name: str = None):
    """Import Gaussian component of Zeta field.

    """

    if not file_name:
        file_name = filenames[3]

    # Gaussian zeta field
    zeta_g_file = fields_path/file_name
    in_zeta_g   = open(zeta_g_file, 'rb')
    # Read in zeta_g, reshape it into an nxnxn, and then trim off buffers
    zeta_g = np.fromfile(in_zeta_g,dtype=np.float32,count=-1)
    zeta_g = np.reshape(zeta_g, (l_array,l_array,l_array), order='F')
    zeta_g = zeta_g[l_buff:-l_buff,l_buff:-l_buff,l_buff:-l_buff]

    return zeta_g

def get_zeta_ng(z_file_name: str = None, zg_file_name: str = None):
    """Import nonG component of Zeta (zeta - zeta_g = zeta_ng).

    """

    # nonG component of Zeta
    zeta_ng = get_zeta(z_file_name) - get_zeta_g(zg_file_name)

    return zeta_ng

def get_zeta_all(z_file_name: str = None, zg_file_name: str = None):
    """Import Zeta fields
    
    """
    
    print('\nProcessing Zeta fields/components...\n')
    zeta = get_zeta(z_file_name)
    zeta_g = get_zeta_g(zg_file_name)
    zeta_ng = get_zeta_ng(z_file_name, zg_file_name)

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

def main(path_realization: str | Path = None):
    """Main function.

    """

    import_params(path_realization)

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

    return [delta, delta_g, delta_ng, zeta, zeta_g, zeta_ng]

#--------------------------------------------------#

if __name__=="__main__":

    path_realization = Path('peak-patch-runs/n1024bigR/z2/fnl1e6')
    import_params(path_realization)

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