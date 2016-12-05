
# usage
# python kspace_density_synt.py

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

from qmworks.utils import zipWith
from qmworks.common import AtomXYZ

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

def main():
    """
    These calculation is based on the paper:
    `Theoretical analysis of electronic band structure of
    2- to 3-nm Si nanocrystals`
    PHYSICAL REVIEW B 87, 195420 (2013)
    """
    
    
    # ======= prepare system
    
    natoms = 20
    lattice_cte = 4.0
    coords   = np.array( [ [i,0.0,0.0] for i in range(natoms) ] ) * lattice_cte
    symbols  = ['Si'] * natoms
    
    #atoms = [(symbol,xyz) for symbol,xyz in zip(symbols,coords) ]
    atoms     = zipWith(AtomXYZ)(symbols)(coords)
    path_hdf5 = os.path.join( data_dir, "quantum.hdf5" )
    dictCGFs  = create_dict_CGFs(path_hdf5, "DZVP-MOLOPT-SR-GTH", atoms)
    
    cgfs = dictCGFs[ 'Si' ]
    print( "cgfs", cgfs )
    print( "len(cgfs) : ", len(cgfs) )
    
    orbTypes = [ p.orbType for p in cgfs ]
    
    per_atom_coefs =  np.array( [ 1.0, 0.0,   0.0, 0.0, 0.0,    0.0, 0.0, 0.0,      0.0, 0.0, 0.0, 0.0, 0.0, 0.0  ] )
    
    #per_atom_coefs =  np.array( [ 0.0, 0.0,   1.0, 0.0, 0.0,    0.0, 0.0, 0.0,      0.0, 0.0, 0.0, 0.0, 0.0, 0.0  ] )
    #per_atom_coefs =  np.array( [ 0.0, 0.0,   0.0, 0.0, 0.0,    0.0, 0.0, 0.0,      1.0, 0.0, 0.0, 0.0, 0.0, 0.0  ] )
    
    
    #mo_i = per_atom_coefs * natoms
    
    #mo_i  = np.concatenate( [ per_atom_coefs for i in range(natoms) ] )
    mo_i  = np.concatenate( [ per_atom_coefs*((-1)**i) for i in range(natoms) ] )
    
    
    
    print( mo_i )
   
    # ======= real space
    
    XYZs, indBounds, (amin,amax,aspan) = rwf.pre_wf_real( coords, Rcut = 6.0, dstep=np.array((0.5,0.5,0.5)) )
    wf = rwf.wf_real( symbols, coords, dictCGFs, mo_i, XYZs, indBounds )
    
    lvec=np.array( [[0,0,0],[aspan[0]/angstroms_to_au,0,0],[0,aspan[1]/angstroms_to_au,0],[0,0,aspan[2]/angstroms_to_au]] )
    rwf.saveXSF( "wf_synt.xsf", np.transpose( wf, (2,1,0) ), lvec, symbols=symbols, coords=(coords-amin[None,:])/angstroms_to_au ) 

    lvec=np.array( [[0,0,0],[ 10*pi*angstroms_to_au/aspan[0],0,0],[0,10*pi*angstroms_to_au/aspan[1],0],[0,0,10*pi*angstroms_to_au/aspan[2]]] )
    wfk  = np.fft.fftn    (wf)
    wfk  = np.fft.fftshift(wfk)
    rhok = np.absolute    (wfk) 
    rwf.saveXSF( "wfk_synt.xsf", np.transpose( rhok, (2,1,0) ), lvec ) 
    
    plt.subplot(2,1,1);
    plt.plot( range(rhok.shape[0]), rhok[:,rhok.shape[1]/2,rhok.shape[2]/2] )
    

    # ======= k-space
        
    # K-space grid to calculate the fuzzy band
    nPoints = 1000
    lscale  = 3*np.pi/lattice_cte
    clin    = np.linspace(0.0,1.0,nPoints)[:,None]
    

    klines = [ ((-1.0, 0.0, 0.0),(1.0,0.0,0.0)) ]
    
    
    for kline in klines:
        kmin      = np.array(kline[0]); kmax = np.array(kline[1]);
        kpoints   = kmin[None,:]*(1-clin) + clin*kmax[None,:]
        kpoints  *= lscale    
        chikdic   = get_fourier_basis( symbols, dictCGFs, kpoints )
        orbK      = calculate_fourier_trasform_cartesian( symbols, coords, dictCGFs, mo_i, kpoints, chikdic=chikdic )
        orbKdens  = np.absolute( orbK )
        plt.subplot(2,1,2);
        plt.plot( clin, orbKdens )
        
    plt.show()    
    
    
if __name__ == "__main__":
    main()
