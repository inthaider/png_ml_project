## !/usr/bin/env python3
"""Imports relevant Peak-Patch parameters for a given field.

@Authors:   Jibran Haider

Examples
--------
TESTI'm currently setup to run in terminal with a command::

    $ python3 get_params.py

Or you can run me in a .ipynb notebook with::

    %run get_params.py

Or you can import me as a module::

    import get_params

Or you can import relevant attributes (variables) using::

    from get_params import ...

Attributes
----------
fields_path : 

l_array : 

l_mpc : 

l_buff : 

l_trim : 

Notes
-----
TODO: In final implementation, will have to take path to ppatch runs as input.

Example Peak-Patch realization: 
    peak-patch-runs/n1024bigR/z2/fnl1e6

"""

#----Import modules----#

# Include standard modules
import sys
import os

import argparse
from importlib.util import spec_from_loader, module_from_spec # For path manipulations and module loading
from importlib.machinery import SourceFileLoader # For path manipulations and module loading
from pathlib import Path # For path manipulations and module loading
import math
from sysconfig import get_path
from types import ModuleType

import numpy as np

#----Declare variables----#

path_pkp_realization: str | Path = None
path_fields: str | Path = None
params: ModuleType
full_name: str = ''
l_mpc: int | float = 0.0
l_array: int = 0
l_buff: int = 0
l_trim: int = 0

#--------------------------------------------------#

def set_path_realization(path_realization: str | Path = None):
    """Set the path to a given Peak-Patch realization.
    
    """

    global path_pkp_realization

    # Check for path
    if path_realization == None:
        # Initiate parser
        parser = argparse.ArgumentParser()
        
        # Add argument
        parser.add_argument("--path", "-pth", type=str, nargs="*", help="Set path to Peak-Patch realization.")
        
        # Read arguments from the command lines
        # print(argparse.Namespace)
        args, unknown = parser.parse_known_args()

        if args.path:
            path_realization = args.path
        else:
            path_realization = input_path_realization()

    path_realization = Path(path_realization)    
    path_pkp_realization = path_realization
        
    return

def get_path_realization():
    """Get the stored path to a Peak-Patch realization.
    
    """
    
    return Path(path_pkp_realization)

def input_path_realization():
    """Take user-input for the path to a given Peak-Patch realization.
    
    """

    path_realization = input("Enter the path for your Peak-Patch realization (abs or rel; no quotes): ")
    assert os.path.exists(path_realization), "I did not find the file at, " + str(path_realization)
    print("Hooray, we found your path!")

    path_realization = Path(path_realization) 

    return path_realization

def get_path_params(path_realization: str | Path = None):
    """Get path to parameters' file (named 'param.params' in pkpatch).

    """

    if path_realization == None:
        path_realization = get_path_realization()
    
    # Path to params file
    path_params = Path(path_realization)/'param/param.params'

    return path_params

def get_path_fields(path_realization: str | Path = None):
    """Get path to fields' dir.

    """

    global path_fields

    if path_realization == None:
        path_realization = get_path_realization()
    
    # Path to dir of relevant field realizations
    path_fields = Path(path_realization)/'fields'

    return path_fields

def load_module(path_params):
    """This piece of code loads the parameters file, param.params. 

    Needed to do it this way because of multiple subdirectories in the tree and the period/dot in the name "param.params".
    Would have had to create __init__.py files in each subdir otherwise.

    Got it from: 
    https://csatlas.com/python-import-file-module/#import_any_file_including_non_py_file_extension_python_3_4_and_up

    """

    global params

    loader = SourceFileLoader( 'param.params', str(path_params))
    spec = spec_from_loader( 'param.params', loader )
    params = module_from_spec( spec )
    loader.exec_module( params )

    return params

def set_module_name(parameters):
    """Save the 'params' module with its full name containing most important params.

    sys.modules[] saves the module "params" with its full name containing the most imp. params.
    This allows importing "params" directly with the full name.
    The full name is "SideLengthInMpc_SideLengthOfArray_BufferThickness_?".

    """

    global full_name

    full_name = f'params_{math.floor(parameters.boxsize)}Mpc_n{parameters.nmesh}_nb{parameters.nbuff}_nt{1}'
    sys.modules[full_name] = parameters

    return full_name

#--------------------------------------------------#

def get_params(path_realization: str | Path = None, save_module=False):
    """Import relevant parameters needed.



    Returns:
    --------
    boxsize : 
        Side length of the cubic simulation volume (Mpc/h)
    nmesh : int
        Side length of field array (pixels/array units)
    nbuff : int
        Buffer thickness (pixels/array units)

    *Note on Mpc/h:
    An indeterminate unit of distance between 4/3 Mpc and 2 Mpc. 
    The h is a parameter in the closed interval [0.5, 0.75] and reflects an uncertainty in the Hubble constant. 
    For example, in a universe where the Hubble constant is 70km/Mpc/s, h is 0.7. 
    Similarly, a Hubble constant of 50km/Mpc/s would lead to a value for h of 0.5.*
    
    """

    global l_mpc, l_array, l_buff, l_trim

    if path_realization == None:
        path_realization = get_path_realization()
    path_params = get_path_params(path_realization)
    parameters = load_module(path_params)

    # Side length of the cubic simulation volume
    l_mpc = parameters.boxsize #Mpc/h
    # Side length of field arrays
    l_array = parameters.nmesh
    # Buffer thickness
    l_buff = parameters.nbuff
    # Side length of field array after trimming buffers
    l_trim = l_array - l_buff * 2

    return (l_mpc, l_array, l_buff, l_trim, set_module_name(parameters)) if save_module == True else (l_mpc, l_array, l_buff, l_trim)

#--------------------------------------------------#

def main(path_realization: str | Path = None, save_module=False):
    """Main function.
    
    """
    
    set_path_realization(path_realization)
    output = get_params(path_realization, save_module) + (get_path_fields(), )

    l_mpc, l_array, l_buff, l_trim = output[0], output[1], output[2], output[3]
    print('Side length (Mpc/h): ', l_mpc, 
            '\nSide length (array units): ', l_array, 
            '\nBuffer thickness (array units): ', l_buff,
            '\nSide length after trimming buffers: ', l_trim)

    ### This is for which realization of initial fields? ###

    return output


if __name__=="__main__":
    # # Get path to current file's dir
    # current_path = Path().absolute()
    # # Path to relevant realization
    # path_pkp_realization = current_path/'peak-patch-runs/n1024bigR/z2/fnl1e6/'
    # peak-patch-runs/n1024bigR/z2/fnl1e6/

    set_path_realization()
    l_mpc, l_array, l_buff, l_trim, module_name = get_params(save_module=True)
    path_fields = get_path_fields()

    ### This is for which realization of initial fields? ###

    print('Side length (Mpc/h): ', l_mpc, 
            '\nSide length (array units): ', l_array, 
            '\nBuffer thickness (array units): ', l_buff,
            '\nSide length after trimming buffers: ', l_trim,
            '\nModule saved with name:', module_name)

#--------------------------------------------------#

#########################################################################
#----Potentially useful code:----#
#########################################################################
# ppatchruns_dir = os.path.join( script_dir, '..', 'alpha', 'beta' ) # Joins different paths together
# Path("path/to/file.txt").touch() # Creates new empty file.txt (can create other files similarly)
#-----------#
## Get path to current file's parent dir
# parent_path = current_path.parent
#########################################################################