
# ==== Definitions
# usage: python real_space_wf.py -p ethenol -hdf5 ethenol.hdf5 -xyz ethenol.xyz -lower 1 -upper 39

import sys
sys.path.append('/home/prokop/git_SW/nonAdiabaticCoupling')

#data_dir = "/home/prokop/Desktop/kscan_qmworks/Si68-H"
#data_dir = "/home/prokop/Desktop/kscan_qmworks/Si538H"
data_dir = "/home/prokop/Desktop/kscan_qmworks/molecule"

from functools import (partial, reduce)
from math import (pi, sqrt)
from multiprocessing import Pool
from nac.schedule.components        import create_dict_CGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ
from qmworks.utils import concat

import nac.integrals.realSpaceWf as rwf

from nac import retrieve_hdf5_data
import h5py

import matplotlib.pyplot as plt

import argparse
import itertools
import numpy as np
import os



from typing import (Callable, List, Tuple)
Vector = np.ndarray
Matrix = np.ndarray

angstroms_to_au = 1.889725989

# ==== Functions

def print_attrs(name, obj):
    print( name )

def main(parser):
    """
    These calculation is based on the paper:
    `Theoretical analysis of electronic band structure of
    2- to 3-nm Si nanocrystals`
    PHYSICAL REVIEW B 87, 195420 (2013)
    """
    project_name, path_hdf5, path_xyz, basis_name, lower, upper = read_cmd_line(parser)
    path_hdf5 = join( data_dir, path_hdf5 )
    
    f = h5py.File(path_hdf5,'r')
    f.visititems(print_attrs)
    path_energy = project_name+'/point_0/cp2k/mo/eigenvalues'
    path_coefs  = project_name+'/point_0/cp2k/mo/coefficients'
    Es = retrieve_hdf5_data( path_hdf5, [path_energy])[0]
    print( "len(Es)", len(Es) )
    print( "Es =", np.array(Es)*27.2114 )
    
    path_xyz        = join( data_dir, path_xyz )
    atoms           = readXYZ(path_xyz)
    symbols         = np.array([at.symbol for at in atoms])
    coords_angstrom = np.array([at.xyz    for at in atoms])
    coords          = angstroms_to_au * coords_angstrom
    
    XYZs, indBounds, (amin,amax,aspan) = rwf.pre_wf_real( coords, Rcut = 6.0, dstep=np.array((0.5,0.5,0.5)) )
    
    dictCGFs        = create_dict_CGFs(path_hdf5, basis_name, atoms)
    
    orbitals = range(lower, upper)
    for orbital in orbitals:
        print ( " orbital ", orbital )
        mo_i = retrieve_hdf5_data(path_hdf5, path_coefs )[:, orbital]
        wf = rwf.wf_real( symbols, coords, dictCGFs, mo_i, XYZs, indBounds )
        
        print( "wf.shape ", wf.shape )
        
        #lvec=np.array( [amin,[aspan[0],0,0],[0,aspan[1],0],[0,0,aspan[2]]] )    
        lvec=np.array( [[0,0,0],[aspan[0]/angstroms_to_au,0,0],[0,aspan[1]/angstroms_to_au,0],[0,0,aspan[2]/angstroms_to_au]] )
        rwf.saveXSF( "wf_%0i.xsf" %orbital, np.transpose( wf, (2,1,0) ), lvec, symbols=symbols, coords=(coords-amin[None,:])/angstroms_to_au )    
        
        #wfk  = np.fft.fftn( wf )
        #wfk  = np.fft.fftshift(wfk)
        #rhok = np.absolute( wfk ) 
        #saveXSF( "wfk_%0i.xsf" %orbital, rhok, lvec ) 

    # Dictionary containing as key the atomic symbols and as values the set of CGFs

def read_cmd_line(parser):
    args = parser.parse_args()
    attributes = ['p', 'hdf5', 'xyz', 'basis', 'lower', 'upper']
    return [getattr(args, x) for x in attributes]


if __name__ == "__main__":
    msg = " script -hdf5 <path/to/hdf5> -xyz <path/to/geometry/xyz -alat lattice_cte -b basis_name"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p',     help='Project name'             , required=True)
    parser.add_argument('-hdf5',  help='path to the HDF5 file'    , required=True)
    parser.add_argument('-xyz',   help='path to molecular gemetry', required=True)
    parser.add_argument('-basis', help='Basis Name'                              ,default="DZVP-MOLOPT-SR-GTH")
    parser.add_argument('-lower', help='lower orbitals to compute band', type=int, default=19)
    parser.add_argument('-upper', help='upper orbitals to compute band', type=int, default=21)

    main(parser)
