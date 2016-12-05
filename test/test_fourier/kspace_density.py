

'''

usage:
python kspace_density.py -p Si538H240 -hdf5 quantum.hdf5 -xyz Si538H240.xyz -alat 5.40 -lower 1 -upper 39

'''


import sys
sys.path.append('/home/prokop/git_SW/nonAdiabaticCoupling')
data_dir = "/home/prokop/Desktop/kscan_qmworks/Si538H"

from functools import (partial, reduce)
from math import (pi, sqrt)
from multiprocessing import Pool
from nac.integrals.fourierTransform import calculate_fourier_trasform_cartesian, get_fourier_basis
from nac.schedule.components        import create_dict_CGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ
from qmworks.utils import concat

from nac import retrieve_hdf5_data
import h5py

import matplotlib.pyplot as plt

import argparse
import itertools
import numpy as np
import os

# Some Hint about the types
from typing import (Callable, List, Tuple)
Vector = np.ndarray
Matrix = np.ndarray

angstroms_to_au = 1.889725989

klines_glob = [
    (( 2.0,  0.0,  0.0),  ( 1.0,  1.0,  0.0)),
    (( 2.0,  0.0,  0.0),  ( 1.0, -1.0,  0.0)), 
    (( 2.0,  0.0,  0.0),  ( 1.0,  0.0,  1.0)), 
    (( 2.0,  0.0,  0.0),  ( 1.0,  0.0, -1.0)),
    (( 0.0,  2.0,  0.0),  ( 1.0,  1.0,  0.0)),
    (( 0.0,  2.0,  0.0),  (-1.0,  1.0,  0.0)),
    (( 0.0,  2.0,  0.0),  ( 0.0,  1.0,  1.0)),
    (( 0.0,  2.0,  0.0),  ( 0.0,  1.0, -1.0)),
    (( 0.0,  0.0,  2.0),  (-1.0,  0.0,  1.0)),    
    (( 0.0,  0.0,  2.0),  ( 1.0,  0.0,  1.0)),  
    (( 0.0,  0.0,  2.0),  ( 0.0,  1.0,  1.0)),  
    (( 0.0,  0.0,  2.0),  ( 0.0, -1.0,  1.0)),  
    ((-2.0,  0.0,  0.0),  (-1.0,  1.0,  0.0)),
    ((-2.0,  0.0,  0.0),  (-1.0, -1.0,  0.0)), 
    ((-2.0,  0.0,  0.0),  (-1.0,  0.0,  1.0)), 
    ((-2.0,  0.0,  0.0),  (-1.0,  0.0, -1.0)), 
    (( 0.0, -2.0,  0.0),  ( 1.0, -1.0,  0.0)),
    (( 0.0, -2.0,  0.0),  (-1.0, -1.0,  0.0)),
    (( 0.0, -2.0,  0.0),  ( 0.0, -1.0,  1.0)),
    (( 0.0, -2.0,  0.0),  ( 0.0, -1.0, -1.0)),
    (( 0.0,  0.0, -2.0),  ( 1.0,  0.0, -1.0)),    
    (( 0.0,  0.0, -2.0),  (-1.0,  0.0, -1.0)),  
    (( 0.0,  0.0, -2.0),  ( 0.0,  1.0, -1.0)),  
    (( 0.0,  0.0, -2.0),  ( 0.0, -1.0, -1.0)), 
    (( 1.0,  1.0,  1.0),  ( 0.0,  1.0,  1.0)),
    (( 1.0,  1.0,  1.0),  ( 1.0,  0.0,  1.0)),
    (( 1.0,  1.0,  1.0),  ( 1.0,  1.0,  0.0)),
    ((-1.0,  1.0,  1.0),  ( 0.0,  1.0,  1.0)),
    ((-1.0,  1.0,  1.0),  (-1.0,  0.0,  1.0)),
    ((-1.0,  1.0,  1.0),  (-1.0,  1.0,  0.0)),
    (( 1.0, -1.0,  1.0),  ( 0.0, -1.0,  1.0)),
    (( 1.0, -1.0,  1.0),  ( 1.0,  0.0,  1.0)), 
    (( 1.0, -1.0,  1.0),  ( 1.0, -1.0,  0.0)),
    ((-1.0, -1.0,  1.0),  ( 0.0, -1.0,  1.0)),
    ((-1.0, -1.0,  1.0),  (-1.0,  0.0,  1.0)),
    ((-1.0, -1.0,  1.0),  (-1.0, -1.0,  0.0)),
    (( 1.0,  1.0, -1.0),  ( 0.0,  1.0, -1.0)), 
    (( 1.0,  1.0, -1.0),  ( 1.0,  0.0, -1.0)), 
    (( 1.0,  1.0, -1.0),  ( 1.0,  1.0,  0.0)), 
    ((-1.0,  1.0, -1.0),  ( 0.0,  1.0, -1.0)), 
    ((-1.0,  1.0, -1.0),  (-1.0,  0.0, -1.0)), 
    ((-1.0,  1.0, -1.0),  (-1.0,  1.0,  0.0)), 
    (( 1.0, -1.0, -1.0),  ( 0.0, -1.0, -1.0)),
    (( 1.0, -1.0, -1.0),  ( 1.0,  0.0, -1.0)),
    (( 1.0, -1.0, -1.0),  ( 1.0, -1.0,  0.0)),
    ((-1.0, -1.0, -1.0),  ( 0.0, -1.0, -1.0)),
    ((-1.0, -1.0, -1.0),  (-1.0,  0.0, -1.0)), 
    ((-1.0, -1.0, -1.0),  (-1.0, -1.0,  0.0)), 
    ]

# ============ Functions

def projectionsToBins( kdata, Es, Emin=-6.0, Emax=0.0, dE=0.02 ):
    nks = kdata.shape[1]
    nE   = int((Emax-Emin)/dE) 
    bins = np.zeros( (nE, nks) )
    for i,Ei in enumerate(Es):
        iE = int((Ei-Emin)/dE)
        if (iE>0) and (iE<nE):
            bins[iE,:] = np.maximum( bins[iE,:], kdata[i,:] )
    extent=(0,1,Emin,Emax)
    return bins, extent

def print_attrs(name, obj):
    print( name )

def main(parser):
    """
    These calculation is based on the paper:
    `Theoretical analysis of electronic band structure of
    2- to 3-nm Si nanocrystals`
    PHYSICAL REVIEW B 87, 195420 (2013)
    """
    
    project_name, path_hdf5, path_xyz, lattice_cte, basis_name, lower, upper = read_cmd_line(parser)
    path_hdf5 = join( data_dir, path_hdf5 )
    
    f = h5py.File(path_hdf5,'r')
    #f.visititems(print_attrs)
    path_energy = project_name+'/point_0/cp2k/mo/eigenvalues'
    path_coefs  = project_name+'/point_0/cp2k/mo/coefficients'
    Es = retrieve_hdf5_data( path_hdf5, [path_energy])[0]
    print( "len(Es)", len(Es) )
    print( "Es =", np.array(Es)*27.2114 )
    
    atoms           = readXYZ( join( data_dir, path_xyz ) )
    symbols         = np.array( [at.symbol for at in atoms] )
    coords_angstrom = np.array( [at.xyz    for at in atoms] )

    coords          = angstroms_to_au * coords_angstrom
    lattice_cte     = lattice_cte * angstroms_to_au

    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms)

    nPoints = 40

    clin = np.linspace( 0.0, 1.0, nPoints )[:,None]
    kpoints = []
    klines = np.array(klines_glob)
    for kline in klines:
        kpoints.append( (1-clin)*kline[0][None,:] + clin*kline[1][None,:] )
    kpoints  = np.concatenate( kpoints )
    kpoints *= (1/lattice_cte)
    print ( "kpoints.shape ", kpoints.shape )
    
    orbitals = list(range(lower, upper + 1))
    dim_x    = len(orbitals)
    result   = np.empty((dim_x, nPoints))
    
    print ("building basiset fourier dictionary ... ")
    chikdic = get_fourier_basis( symbols, dictCGFs, kpoints )
    print ("...fourier basis DONE !")
    for i, orb in enumerate(orbitals):
        print("Orbital: ", orb)
        mo_i      = retrieve_hdf5_data(path_hdf5, path_coefs )[:,orb]
        orbK      = calculate_fourier_trasform_cartesian( symbols, coords, dictCGFs, mo_i, kpoints, chikdic=chikdic )
        orbK      = np.absolute( orbK )
        orbK      = orbK.reshape( len(klines), -1 )
        print ("orbK.shape", orbK.shape)
        result[i] = np.sum( orbK, axis=0 )
    
    Es = [ Es[i]*27.2114 for i in orbitals ]
    
    bins, extent = projectionsToBins( result, Es, Emin=-6.0, Emax=0.0, dE=0.02 )
    
    plt.figure(figsize=(3,8))
    plt.imshow( np.log10(bins), interpolation='nearest', origin='image', extent=extent, cmap='jet' )
    plt.colorbar()
    plt.savefig( "kvaziband.png", bbox='tight' )
    plt.show()    
    
def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    attributes = ['p', 'hdf5', 'xyz', 'alat', 'basis', 'lower', 'upper']

    return [getattr(args, x) for x in attributes]

if __name__ == "__main__":
    msg = " script -hdf5 <path/to/hdf5> -xyz <path/to/geometry/xyz -alat lattice_cte -b basis_name"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True, help='Project name')
    parser.add_argument('-hdf5', required=True, help='path to the HDF5 file')
    parser.add_argument('-xyz', required=True,
                        help='path to molecular gemetry')
    parser.add_argument('-alat', required=True, help='Lattice Constant [Angstroms]',
                        type=float)
    parser.add_argument('-basis', help='Basis Name',
                        default="DZVP-MOLOPT-SR-GTH")
    parser.add_argument('-lower',
                        help='lower orbitals to compute band', type=int,
                        default=19)
    parser.add_argument('-upper',
                        help='upper orbitals to compute band', type=int,
                        default=21)

    main(parser)
