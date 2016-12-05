
__all__ = ["calculate_fourier_trasform_cartesian", "get_fourier_basis"]

from cmath import pi
from nac.common import retrieve_hdf5_data
from os.path import join

import numpy as np

# Some Hints about the types
from typing import (Dict, List, NamedTuple, Tuple)
Vector = np.ndarray
Matrix = np.ndarray

def get_fourier_basis( atomic_symbols: Vector, dictCGFs: Dict, kpoints: Vector ):
    '''
    for each element in system compute Fourier transfroms of its basis functions in given set of k-points
    and store them into dictionary 
    '''
    unique_symbols = {symbol: None for symbol in atomic_symbols}
    xyz0 = np.zeros(3)
    for symbol in unique_symbols:
        cgfs = dictCGFs[symbol]
        chik = np.zeros( (len(cgfs),len(kpoints)), dtype=np.complex128 )
        chik = calculate_fourier_trasform_atom( kpoints, cgfs )
        unique_symbols[symbol] = chik
    return unique_symbols

def calculate_fourier_trasform_cartesian(symbols: Vector,
                                         coords: Vector,
                                         dictCGFs: Dict,
                                         mo_i : Vector,
                                         kpoints: Vector,
                                         chikdic=None) -> Vector:
    """
    Calculate the Fourier transform projecting the MO in a set of plane waves
    mo_fourier(k) = < phi(r) | exp(i k . r)>
    """
    if (chikdic is None):
        chikdic = get_fourier_basis( symbols, dictCGFs, kpoints )
    
    result = np.zeros( len(kpoints), dtype=np.complex128 )
    acc = 0
    for symbol, xyz in zip(symbols, coords ):
        krs = -2j*pi*np.dot( kpoints, xyz )
        eikrs = np.exp( krs )
        cfgks = chikdic[symbol]
        dim_cgfs = len(cfgks)
        coefs = mo_i[acc:acc+dim_cgfs]
        prod  = coefs[:,None]*cfgks*eikrs[None,:]
        result += np.sum( prod, axis=0 ) 
        acc += dim_cgfs
    return result

def calculate_fourier_trasform_atom(ks: Vector, cgfs: List)-> Vector:
    """
    Compute 3D Fourier transform for basis function of particular element for set of kpoints "ks"
    """
    arr = np.empty((len(cgfs), len(ks)), dtype=np.complex128)
    for i, cgf in enumerate(cgfs):
        arr[i] = calculate_fourier_trasform_contracted(cgf, ks)
    return arr

def calculate_fourier_trasform_contracted(cgf: NamedTuple, ks: Vector) -> complex:
    """
    Compute 3D Fourier transform for given basis function "cgf" composed of gaussian primitives for set of kpoints "ks"
    """
    cs, es = cgf.primitives
    label = cgf.orbType
    angular_momenta = compute_angular_momenta(label)
    res = np.zeros( len(ks), dtype=np.complex128)
    for c,e in zip( cs, es ):
        fouKx = calculate_fourier_trasform_primitive(angular_momenta[0], ks[:,0], e)
        fouKy = calculate_fourier_trasform_primitive(angular_momenta[1], ks[:,1], e)
        fouKz = calculate_fourier_trasform_primitive(angular_momenta[2], ks[:,2], e)
        res +=c*( fouKx * fouKy * fouKz )    
    return res
    
def calculate_fourier_trasform_primitive(l: int, ks: Vector, alpha: float) -> complex:
    '''
    calculate 1D Fourier transform of single gausian primitive function centred in zero
     for given ste of kpoints "ks"
    '''
    gauss = np.exp(-(pi**2/alpha) * ks**2)
    if l == 0:
        return np.sqrt(pi/alpha) * gauss
    elif l == 1:
        c = -1j*np.sqrt((pi/alpha)**3) 
        return c * ks * gauss
    elif l == 2:
        c = np.sqrt(pi / alpha**5)
        return c * ((alpha/2.0) - (pi**2)*(ks**2)) * gauss
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)  

def compute_angular_momenta(label) -> Vector:
    """
    Compute the exponents l,m and n for the CGF: x^l y^m z^n exp(-a (r-R)^2)
    """
    orbitalIndexes = {("S", 0): 0, ("S", 1): 0, ("S", 2): 0,
                      ("Px", 0): 1, ("Px", 1): 0, ("Px", 2): 0,
                      ("Py", 0): 0, ("Py", 1): 1, ("Py", 2): 0,
                      ("Pz", 0): 0, ("Pz", 1): 0, ("Pz", 2): 1,
                      ("Dxx", 0): 2, ("Dxx", 1): 0, ("Dxx", 2): 0,
                      ("Dxy", 0): 1, ("Dxy", 1): 1, ("Dxy", 2): 0,
                      ("Dxz", 0): 1, ("Dxz", 1): 0, ("Dxz", 2): 1,
                      ("Dyy", 0): 0, ("Dyy", 1): 2, ("Dyy", 2): 0,
                      ("Dyz", 0): 0, ("Dyz", 1): 1, ("Dyz", 2): 1,
                      ("Dzz", 0): 0, ("Dzz", 1): 0, ("Dzz", 2): 2}
    lookup = lambda i: orbitalIndexes[(label, i)]

    return np.apply_along_axis(np.vectorize(lookup), 0, np.arange(3))

