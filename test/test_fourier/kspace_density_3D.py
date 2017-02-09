# uasage
#  python kspace_density_3D.py -p Si538H240 -hdf5 quantum.hdf5 -xyz Si538H240.xyz -alat=5.40 -lower 1 -upper 39


import sys
sys.path.append('/home/prokop/git_SW/nonAdiabaticCoupling')

#data_dir = "/home/prokop/Desktop/kscan_qmworks/Si68-H"
data_dir = "/home/prokop/Desktop/kscan_qmworks/Si538H"

from functools import (partial, reduce)
from math import (pi, sqrt)
from multiprocessing import Pool
from nac.integrals.fourierTransform import calculate_fourier_trasform_cartesian, get_fourier_basis
import nac.integrals.realSpaceWf as rwf
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

# ============ Functions

def projectionsToBins( kdata, Es, Emin=-6.0, Emax=0.0, dE=0.02 ):
    nks = kdata.shape[1]
    nE   = int((Emax-Emin)/dE) 
    bins = np.zeros( (nE, nks) )
    for i,Ei in enumerate(Es):
        iE = int((Ei-Emin)/dE)
        if (iE>0) and (iE<nE):
            bins[iE,:] = np.maximum( bins[iE,:], kdata[i,:] )
    extent=(0,2,Emin,Emax)
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
    # Parse Command line
    project_name, path_hdf5, path_xyz, lattice_cte, basis_name, lower, \
        upper = read_cmd_line(parser)
    # Coordinates transformation
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
    
    #print( atoms ); exit()
    
    symbols         = np.array([at.symbol for at in atoms])
    coords_angstrom = np.array([at.xyz for at in atoms])

    coords          = angstroms_to_au * coords_angstrom
    lattice_cte     = lattice_cte * angstroms_to_au

    # Dictionary containing as key the atomic symbols and as values the set of CGFs
    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms)
    count_cgfs = np.vectorize(lambda s: len(dictCGFs[s]))
    number_of_basis = np.sum(np.apply_along_axis(count_cgfs, 0, symbols))

    orbital = 20
    mo_i = retrieve_hdf5_data(path_hdf5, path_coefs )[:, orbital]
    
    
    XYZs, indBounds, (amin,amax,aspan) = rwf.pre_wf_real( coords, Rcut = 6.0, dstep=np.array((0.5,0.5,0.5)) )
    wf = rwf.wf_real( symbols, coords, dictCGFs, mo_i, XYZs, indBounds )
    
    lvec=np.array( [[0,0,0],[aspan[0]/angstroms_to_au,0,0],[0,aspan[1]/angstroms_to_au,0],[0,0,aspan[2]/angstroms_to_au]] )
    rwf.saveXSF( "wf_%0i.xsf" %orbital, np.transpose( wf, (2,1,0) ), lvec, symbols=symbols, coords=(coords-amin[None,:])/angstroms_to_au ) 

    wfk  = np.fft.fftn    (wf)
    wfk  = np.fft.fftshift(wfk)
    rhok = np.absolute    (wfk) 
    rwf.saveXSF( "wfk_%0i.xsf" %orbital, np.transpose( rhok, (2,1,0)) , lvec ) 
    
    
    # K-space grid to calculate the fuzzy band
    nPoints = 50
    lscale  = 1.0/lattice_cte
    
    kmin  = np.array( [-2.0,-2.0,-2.0] ) * lscale
    kmax  = np.array( [ 2.0, 2.0, 2.0] ) * lscale 
    kspan = kmax - kmin 
    
    print( "kmin", kmin, "kmax", kmax )
               
    k_coords = np.array( [ 
       #  X (1,0,0)
        [ 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],  
        [ 0.0, 1.0, 0.0], [ 0.0,-1.0, 0.0],   
        [ 0.0, 0.0, 1.0], [ 0.0, 0.0,-1.0],
       #  Gamma (2,0,0)
        [ 2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], 
        [ 0.0, 2.0, 0.0], [ 0.0,-2.0, 0.0],   
        [ 0.0, 0.0, 2.0], [ 0.0, 0.0,-2.0],     
       #  Gamma (1,1,1)    
        [ 1.0, 1.0, 1.0], [-1.0,-1.0,-1.0],
        [-1.0, 1.0, 1.0], [ 1.0,-1.0,-1.0],
        [ 1.0,-1.0, 1.0], [-1.0, 1.0,-1.0],
        [ 1.0, 1.0,-1.0], [-1.0,-1.0, 1.0],
       #  X  (1,1,0) 
        [ 1.0, 1.0, 0.0], [-1.0,-1.0, 0.0],
        [ 1.0, 0.0, 1.0], [-1.0, 0.0,-1.0],
        [ 0.0, 1.0, 1.0], [ 0.0,-1.0,-1.0],  
        [ 1.0,-1.0, 0.0], [-1.0, 1.0, 0.0],
        [ 1.0, 0.0,-1.0], [-1.0, 0.0, 1.0],
        [ 0.0, 1.0,-1.0], [ 0.0,-1.0, 1.0],
    ]) * lscale
    
    #k_symbols = ['H']*6 + ['He']*6 + ['Li']*8 + ['Be']*12 
    k_symbols = ['He']*6 + ['Ne']*6 + ['Ar']*8 + ['Kr']*12 
    
    print( k_symbols )

    k_coords = ( k_coords - kmin[None,:] )
    
    print( "k_coords = ", k_coords )
    
    kgridX,kgridY,kgridZ = np.mgrid[:nPoints,:nPoints,:nPoints]
    
    kgridX = kmin[0] + (kspan[0]/kgridX.shape[0]) * (kgridX+0.5)
    kgridY = kmin[1] + (kspan[1]/kgridX.shape[1]) * (kgridY+0.5)
    kgridZ = kmin[2] + (kspan[2]/kgridX.shape[2]) * (kgridZ+0.5)
   
    kpoints = np.stack( [kgridX,kgridY,kgridZ] )
    kpoints = np.transpose( kpoints, (1,2,3,0) )
    
    print( "kpoints.shape", kpoints.shape )
    kpoints = kpoints.reshape(nPoints**3,3) #.copy()
    print( "kpoints.shape", kpoints.shape )
       
    chikdic   = get_fourier_basis( symbols, dictCGFs, kpoints )
    orbK      = calculate_fourier_trasform_cartesian( symbols, coords, dictCGFs, mo_i, kpoints, chikdic=chikdic )
    orbKdens  = np.absolute( orbK )
    
    
    
    orbKdens = orbKdens.reshape(nPoints,nPoints,nPoints)
    print( orbKdens.shape )
    #lvec = [[0.0,0.0,0.0],  [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]
    #lvec = [kmin, [kspan[0],0.0,0.0], [0.0,kspan[1],0.0], [0.0,0.0,kspan[2]] ]
    lvec = [[0.0,0.0,0.0], [kspan[0],0.0,0.0], [0.0,kspan[1],0.0], [0.0,0.0,kspan[2]] ]
    rwf.saveXSF( "orbK_%0i.xsf" %orbital, np.transpose(orbKdens,(2,1,0)), lvec, symbols=k_symbols, coords=k_coords  ) 
        
  
def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    attributes = ['p', 'hdf5', 'xyz', 'alat', 'basis', 'lower', 'upper']

    return [getattr(args, x) for x in attributes]



def fun_density_real(function: Callable, k: float) -> float:
    """ Compute the momentum density"""
    return np.absolute(function(k))


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
