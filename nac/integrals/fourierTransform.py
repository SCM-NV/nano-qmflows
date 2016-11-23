
__all__ = ["calculate_fourier_trasform_cartesian","calculate_fourier_trasform_cartesian_prokop", "get_fourier_basis"
           "real_to_reciprocal_space"]

from cmath import (exp, pi, sqrt)
from functools import partial
from nac.common import retrieve_hdf5_data
from os.path import join

import numpy as np


# Some Hint about the types
from typing import (Dict, List, NamedTuple, Tuple)
Vector = np.ndarray
Matrix = np.ndarray

#================ Prokop

def get_fourier_basis( atomic_symbols: Vector, dictCGFs: Dict, kpoints: Vector ):
    unique_symbols = {}
    for symbol in atomic_symbols:
         unique_symbols[symbol] = None
    xyz0 = np.zeros(3)
    for symbol in unique_symbols:
        cgfs = dictCGFs[symbol]
        chik=np.zeros( (len(cgfs),len(kpoints)), dtype=np.complex128 )
        # --- original Fillipe's version
        #for ik,k in enumerate(kpoints):
        #    chis_k = calculate_fourier_trasform_atom( k, cgfs, xyz0)
        #    chik[:,ik] = chis_k
        chik = calculate_fourier_trasform_atom_prokop( kpoints, cgfs )
        unique_symbols[symbol] = chik
    return unique_symbols

def calculate_fourier_trasform_cartesian_prokop(atomic_symbols: Vector,
                                         atomic_coords: Vector,
                                         dictCGFs: Dict,
                                         number_of_basis: int,
                                         path_hdf5: str,
                                         project_name: str,
                                         orbital: int,
                                         kpoints: Vector, 
                                         chikdic =None ) -> Vector:
    if (chikdic is None):
        chikdic = get_fourier_basis( atomic_symbols, dictCGFs, kpoints )
    
    path_to_mo   = join(project_name, 'point_0/cp2k/mo/coefficients')
    mo_i         = retrieve_hdf5_data(path_hdf5, path_to_mo)[:, orbital]
    result = np.zeros( len(kpoints), dtype=np.complex128 )
    stream_coord = chunksOf(atomic_coords, 3)
    acc = 0
    for symbol, xyz in zip(atomic_symbols, stream_coord):
        krs       = 2*-1j*xyz[None,:] * kpoints 
        shiftKs   = np.prod( np.exp( krs ), axis=1 )
        cfgks     = chikdic[symbol]
        dim_cgfs  = len(cfgks)
        coefs     = mo_i[acc:acc+dim_cgfs]
        prod      = coefs[:,None]*cfgks*shiftKs[None,:]
        result   += np.sum( prod, axis=0 ) 
        acc      += dim_cgfs
    return result

def calculate_fourier_trasform_atom_prokop(ks: Vector, cgfs: List)-> Vector:
    """
    Calculate the Fourier transform for the set of CGFs in an Atom.
    """
    arr = np.empty( (len(cgfs),len(ks)), dtype=np.complex128)
    for i, cgf in enumerate(cgfs):
        arr[i] = calculate_fourier_trasform_contracted_prokop(cgf, ks)
    #print( arr )
    #exit()
    return arr

def calculate_fourier_trasform_contracted_prokop(cgf: NamedTuple, ks: Vector) -> complex:
    """
    Compute the fourier transform for a given CGF.
    Implementation note: the function loops over the x,y and z coordinates
    while operate in the whole set of Contracted Gaussian primitives.
    """
    cs, es          = cgf.primitives
    label           = cgf.orbType
    angular_momenta = compute_angular_momenta(label)
    res = np.zeros( len(ks), dtype=np.complex128)
    for c,e in zip( cs, es ):
        fouKx = calculate_fourier_trasform_primitive_prokop( angular_momenta[0], ks[:,0], e )
        fouKy = calculate_fourier_trasform_primitive_prokop( angular_momenta[1], ks[:,1], e )
        fouKz = calculate_fourier_trasform_primitive_prokop( angular_momenta[2], ks[:,2], e )
        #print( fouKx.shape, fouKy.shape, fouKz.shape )
        res +=c*( fouKx * fouKy * fouKz )    
    return res
    
def calculate_fourier_trasform_primitive_prokop(l: int, ks: Vector, alpha: float) -> complex:
    """
    Compute the fourier transform for primitive Gaussian Type Orbitals.
    """
    piks  = pi * ks
    f0   = np.exp(- (piks ** 2) / alpha)
    if l == 0:
        return np.sqrt(pi / alpha) * f0
    elif l == 1:
        c = ((pi / alpha) ** 1.5) * ks * f0
        return 1j * c
    elif l == 2:
        c = np.sqrt(pi / (alpha ** 5))
        return c  * (alpha / 2 - piks ** 2) * f0
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)



#================ ORIGINAL

def calculate_fourier_trasform_cartesian(atomic_symbols: Vector,
                                         atomic_coords: Vector,
                                         dictCGFs: Dict,
                                         number_of_basis: int,
                                         path_hdf5: str,
                                         project_name: str,
                                         orbital: int,
                                         ks: Vector) -> Vector:
    """
    Calculate the Fourier transform projecting the MO in a set of plane waves

    mo_fourier(k) = < phi(r) | exp(i k . r)>

    :param atomic_symbols: Atomic symbols
    :type atomic_symbols: Numpy Array [String]
    :param ks: The vector in k-space where the fourier transform is evaluated.
    :type ks: Numpy Array
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)

    returns: Numpy array
    """
    stream_coord = chunksOf(atomic_coords, 3)
    stream_cgfs = yieldCGF(dictCGFs, atomic_symbols)
    fun = partial(calculate_fourier_trasform_atom, ks)
    molecular_orbital_transformed = np.empty(number_of_basis, dtype=np.complex128)
    acc = 0
    # Fourier transform of the molecular orbitals in Cartesians
    for cgfs, xyz in zip(stream_cgfs, stream_coord):
        dim_cgfs = len(cgfs)
        molecular_orbital_transformed[acc: acc + dim_cgfs] = fun(cgfs, xyz)
        acc += dim_cgfs

    # read molecular orbital
    path_to_mo = join(project_name, 'point_0/cp2k/mo/coefficients')
    mo_i = retrieve_hdf5_data(path_hdf5, path_to_mo)[:, orbital]

    # dot product between the CGFs and the molecular orbitals
    return np.dot(mo_i, molecular_orbital_transformed)


def chunksOf(xs, n):
    """Yield successive n-sized chunks from xs"""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def yieldCGF(dictCGFs, symbols):
    """ Stream of CGFs """
    for symb in symbols:
        yield dictCGFs[symb]


def calculate_fourier_trasform_atom(ks: Vector, cgfs: List,
                                    xyz: Vector)-> Vector:
    """
    Calculate the Fourier transform for the set of CGFs in an Atom.
    """
    arr = np.empty(len(cgfs), dtype=np.complex128)
    for i, cgf in enumerate(cgfs):
        arr[i] = calculate_fourier_trasform_contracted(cgf, xyz, ks)

    return arr


def calculate_fourier_trasform_contracted(cgf: NamedTuple, xyz: Vector,
                                          ks: Vector) -> complex:
    """
    Compute the fourier transform for a given CGF.
    Implementation note: the function loops over the x,y and z coordinates
    while operate in the whole set of Contracted Gaussian primitives.
    """
    cs, es = cgf.primitives
    label = cgf.orbType
    angular_momenta = compute_angular_momenta(label)
    acc = np.ones(cs.shape, dtype=np.complex128)

    # Accumlate x, y and z for each one of the primitives
    for l, x, k in zip(angular_momenta, xyz, ks):
        fun_primitive = partial(calculate_fourier_trasform_primitive, l, x, k)
        rs = np.apply_along_axis(np.vectorize(fun_primitive), 0, es)
        acc *= rs

    # The result is the summation of the primitive multiplied by is corresponding
    # coefficients
    return np.dot(acc, cs)


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


def calculate_fourier_trasform_primitive(l: int, A: float, k: float,
                                         alpha: float) -> complex:
    """
    Compute the fourier transform for primitive Gaussian Type Orbitals.
    """
    shift = exp(- complex(0, 2 * k * A))
    pik = pi * k
    f0 = exp(- (pik ** 2) / alpha)
    if l == 0:
        return shift * sqrt(pi / alpha) * f0
    elif l == 1:
        c = (pi / alpha) ** 1.5 * k * f0
        return complex(0, shift * c)
    elif l == 2:
        c = sqrt(pi / (alpha ** 5))
        return shift * c  * (alpha / 2 - pik ** 2) * f0
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)


# def calculate_fourier_trasform_primitive(l: int, A: float, k: float,
#                                          alpha: float) -> complex:
#     """
#     Compute the fourier transform for primitive Gaussian Type Orbitals.
#     """
#     pik = pi * k
#     f = exp(-alpha * A ** 2 + complex(alpha * A, - pik) ** 2 / alpha)
#     if l == 0:
#         return sqrt(pi / alpha) * f
#     elif l == 1:
#         f = k * exp(-pik * complex(pik / alpha, 2 * A))
#         r = (pi / alpha) ** 1.5  * f
#         return complex(0, -r)
#     elif l == 2:
#         f = exp(-pik * complex(pik / alpha, 2 * A))
#         return sqrt(pi / (alpha ** 5)) * (alpha / 2 - pik ** 2) * f
#     else:
#         msg = ("there is not implementation for the primivite fourier "
#                "transform of l: {}".format(l))
#         raise NotImplementedError(msg)


def real_to_reciprocal_space(tup: Tuple) -> tuple:
    """
    Transform a 3D point from real space to reciprocal space.
    """
    a1, a2, a3 = tup
    cte = 2 * pi / np.dot(a1, cross(a2, a3))

    b1 = cte * cross(a2, a3)
    b2 = cte * cross(a3, a1)
    b3 = cte * cross(a1, a2)

    return b1, b2, b3


def cross(a: Vector, b: Vector) -> Vector:
    """ Cross product"""
    x1, y1, z1 = a
    x2, y2, z2 = b

    x = y1 * z2 - y2 * z1
    y = x2 * z1 - x1 * z2
    z = x1 * y2 - x2 * y1

    return np.array([x, y, z])
