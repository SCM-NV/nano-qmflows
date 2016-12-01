
__all__ = ["calculate_fourier_trasform_cartesian", "get_fourier_basis"]

from cmath import pi
from nac.common import retrieve_hdf5_data
from os.path import join

import numpy as np

# Some Hints about the types
from typing import (Dict, List, NamedTuple, Tuple)
Vector = np.ndarray
Matrix = np.ndarray


# Fourier basis
def get_fourier_basis(atomic_symbols: Vector,
                      dictCGFs: Dict,
                      kpoints: Vector) -> Vector:
    """
    """
    unique_symbols = {symbol: None for symbol in atomic_symbols}
    for symbol in unique_symbols:
        cgfs = dictCGFs[symbol]
        chik = np.zeros((len(cgfs), len(kpoints)), dtype=np.complex128)
        chik = calculate_fourier_trasform_atom(kpoints, cgfs)
        unique_symbols[symbol] = chik
    return unique_symbols


def calculate_fourier_trasform_cartesian(atomic_symbols: Vector,
                                         atomic_coords: Vector,
                                         dictCGFs: Dict,
                                         number_of_basis: int,
                                         path_hdf5: str,
                                         project_name: str,
                                         orbital: int,
                                         kpoints: Vector,
                                         chikdic=None) -> Vector:
    """
    Calculate the Fourier transform projecting the MO in a set of plane waves
    mo_fourier(k) = < phi(r) | exp(i k . r)>
    """
    if (chikdic is None):
        chikdic = get_fourier_basis(atomic_symbols, dictCGFs, kpoints)

    path_to_mo = join(project_name, 'point_0/cp2k/mo/coefficients')
    mo_i = retrieve_hdf5_data(path_hdf5, path_to_mo)[:, orbital]
    result = np.zeros(len(kpoints), dtype=np.complex128)
    stream_coord = chunksOf(atomic_coords, 3)
    acc = 0
    for symbol, xyz in zip(atomic_symbols, stream_coord):
        krs = -2 * 1j * xyz[None, :] * kpoints
        shiftKs = np.prod(np.exp(krs), axis=1)
        cfgks = chikdic[symbol]
        dim_cgfs = len(cfgks)
        coefs = mo_i[acc: acc + dim_cgfs]
        prod = coefs[:, None] * cfgks * shiftKs[None, :]
        result += np.sum(prod, axis=0)
        acc += dim_cgfs
    return result


def calculate_fourier_trasform_atom(ks: Vector, cgfs: List) -> Vector:
    """
    Calculate the Fourier transform for the set of CGFs in an Atom.
    """
    arr = np.empty((len(cgfs), len(ks)), dtype=np.complex128)
    for i, cgf in enumerate(cgfs):
        arr[i] = calculate_fourier_trasform_contracted(cgf, ks)

    return arr


def calculate_fourier_trasform_contracted(cgf: NamedTuple,
                                          ks: Vector) -> complex:
    """
    Compute the fourier transform for a given CGF.
    Implementation note: the function loops over the x,y and z coordinates
    while operate in the whole set of Contracted Gaussian primitives.
    """
    cs, es = cgf.primitives
    label = cgf.orbType
    angular_momenta = compute_angular_momenta(label)
    res = np.zeros(len(ks), dtype=np.complex128)
    for c, e in zip(cs, es):
        ang0 = angular_momenta[0]
        ang1 = angular_momenta[1]
        ang2 = angular_momenta[2]
        fouKx = calculate_fourier_trasform_primitive(ang0, ks[:, 0], e)
        fouKy = calculate_fourier_trasform_primitive(ang1, ks[:, 1], e)
        fouKz = calculate_fourier_trasform_primitive(ang2, ks[:, 2], e)
        res += c * (fouKx * fouKy * fouKz)
    return res


def calculate_fourier_trasform_primitive(l: int, ks: Vector,
                                         alpha: float) -> complex:
    """
    Compute the fourier transform for primitive Gaussian Type Orbitals.
    """
    piks = pi * ks
    f0 = np.exp(-(piks ** 2) / alpha)
    if l == 0:
        return np.sqrt(pi / alpha) * f0
        return np.ones(ks.shape[0])
    elif l == 1:
        c = ((pi / alpha) ** 1.5) * ks * f0
        return 1j * c
        return 0 * ks
    elif l == 2:
        c = np.sqrt(pi / (alpha ** 5))
        return c * (alpha / 2 - piks ** 2) * f0
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)


def chunksOf(xs, n):
    """Yield successive n-sized chunks from xs"""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def yieldCGF(dictCGFs, symbols):
    """ Stream of CGFs """
    for symb in symbols:
        yield dictCGFs[symb]


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
