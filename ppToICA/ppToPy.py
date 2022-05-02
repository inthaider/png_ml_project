# # I'm currently setup to run in terminal with a command:
#
#     python ppToPy.py
#
# This is for a correlated non-gaussianity run using f_NL=10^4.
# Which is a way bigger f_NL than is allowed by the constraints on this
# specific model from experiments, but it's big so it should make the non-
# gaussianity easier to see. All of these fields have are spatial cubes
# with side length s=128 Mpc/h, represented by 


import numpy as np
import sys

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})




# params = './peak_patch_fields/fnl=1e4/param/17Mpc_n1024_nb64_nt2.params'
# append current python modules' folder path
# example: need to import module.py present in '/path/to/python/module/not/in/syspath'
sys.path.append('/Users/JawanHaider/Desktop/research/researchProjects/nonG_project/nonG_code/ppToICA/peak_patch_fields/fnl=1e4/')

import param_params as prm

# params = '/Users/JawanHaider/Desktop/research/researchProjects/nonG_project/nonG_code/ppToICA/peak_patch_fields/fnl=1e4/param/17Mpc_n1024_nb64_nt2.params'
# dict=pykDict.pykDict()
# dict.read_from_file(params)

# Side length of the cubic simulation volume
s = prm.boxsize # in Mpc/h

# Side length of field arrays
n = prm.nmesh

# Buffer thickness
nbuff = prm.nbuff






# Gaussian delta field
delta_g_file = '/Users/JawanHaider/Desktop/research/researchProjects/nonG_project/nonG_code/ppToICA/peak_patch_fields/fnl=1e4/fields/Fvec_17Mpc_n1024_nb64_nt2'
in_delta_g   = open(delta_g_file, 'rb')

# non-Gaussian delta field
delta_file = '/Users/JawanHaider/Desktop/research/researchProjects/nonG_project/nonG_code/ppToICA/peak_patch_fields/fnl=1e4/fields/Fvec_fNL_17Mpc_n1024_nb64_nt2'
in_delta   = open(delta_file, 'rb')


# Gaussian zeta field
zeta_g_file = '/Users/JawanHaider/Desktop/research/researchProjects/nonG_project/nonG_code/ppToICA/peak_patch_fields/fnl=1e4/fields/zetag_17Mpc_n1024_nb64_nt2'
in_zeta_g   = open(zeta_g_file, 'rb')

# non-Gaussian zeta field
zeta_file = '/Users/JawanHaider/Desktop/research/researchProjects/nonG_project/nonG_code/ppToICA/peak_patch_fields/fnl=1e4/fields/zetang_17Mpc_n1024_nb64_nt2'
in_zeta   = open(zeta_file, 'rb')



# Read in delta_g, reshape it into an nxnxn, and then trim off buffers
delta_g = np.fromfile(in_delta_g,dtype=np.float32,count=-1) 
delta_g = np.reshape(delta_g, (n,n,n), order='F')
delta_g = delta_g[nbuff:-nbuff,nbuff:-nbuff,nbuff:-nbuff]

# Read in delta, reshape it into an nxnxn, and then trim off the buffers
delta = np.fromfile(in_delta,dtype=np.float32,count=-1)
delta = np.reshape(delta, (n,n,n), order='F')
delta = delta[nbuff:-nbuff,nbuff:-nbuff,nbuff:-nbuff]

# nonG component of Zeta
delta_ng = delta - delta_g 

# Read in zeta_g, reshape it into an nxnxn, and then trim off buffers
zeta_g = np.fromfile(in_zeta_g,dtype=np.float32,count=-1)
zeta_g = np.reshape(zeta_g, (n,n,n), order='F')
zeta_g = zeta_g[nbuff:-nbuff,nbuff:-nbuff,nbuff:-nbuff]

# Read in zeta, reshape it into an nxnxn, and then trim off the buffers
zeta = np.fromfile(in_zeta,dtype=np.float32,count=-1)
zeta = np.reshape(zeta, (n,n,n), order='F')
zeta = zeta[nbuff:-nbuff,nbuff:-nbuff,nbuff:-nbuff]

# nonG component of Zeta
zeta_ng = zeta - zeta_g 

# # Defines X,Y,Z as meshgrid
# edges = np.linspace( -s/2 , s/2 , n+1 )
# X,Y,Z = np.meshgrid(edges,edges,edges,indexing='ij')

# You now have zeta_g, delta_g, and delta, which are three n-by-n-by-n NumPy arrays representing a gaussian zeta field, a gaussian density field (specifically rho bar times delta, that we talked about today) and a non-gaussian delta field.