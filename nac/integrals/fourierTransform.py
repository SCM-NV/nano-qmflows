
__all__ = ["calculate_fourier_trasform_cartesian", "fun_density_real",
           "real_to_reciprocal_space", "transform_to_spherical"]

from cmath import (exp, pi, sqrt)
from functools import partial
from nac.common import retrieve_hdf5_data
from os.path import join

import numpy as np


# Some Hint about the types
from typing import  Callable, Dict, List, NamedTuple, Tuple
Vector = np.ndarray
Matrix = np.ndarray


def fun_density_real(function: Callable, k: float) -> float:
    """ Compute the momentum density"""
    xs = function(k)
    print("Orbital transformation is: ", xs)
    return np.dot(xs, np.conjugate(xs)).real


def transform_to_spherical(fun_fourier: Callable, path_hdf5: str,
                           project_name: str, orbital: str,
                           k: Vector) -> complex:
    """
    Calculate the Fourier transform in Cartesian, convert it to Spherical
    multiplying by the `trans_mtx` and finally multiply the coefficients
    in Spherical coordinates.
    """
    trans_mtx = retrieve_hdf5_data(path_hdf5, join(project_name, 'trans_mtx'))
    path_to_mo = join(project_name, 'point_0/cp2k/mo/coefficients')
    molecular_orbital_i = retrieve_hdf5_data(path_hdf5, path_to_mo)[:, orbital]

    return np.dot(molecular_orbital_i, np.dot(trans_mtx, fun_fourier(k)))


def calculate_fourier_trasform_cartesian(atomic_symbols: Vector,
                                         atomic_coords: Vector,
                                         dictCGFs: Dict,
                                         number_of_basis: int,
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
    print("K-vector: ", ks)
    stream_coord = chunksOf(atomic_coords, 3)
    stream_cgfs = yieldCGF(dictCGFs, atomic_symbols)
    fun = partial(calculate_fourier_trasform_atom, ks)
    molecular_orbital_transformed = np.empty(number_of_basis, dtype=np.complex128)
    acc = 0
    for cgfs, xyz in zip(stream_cgfs, stream_coord):
        dim_cgfs = len(cgfs)
        molecular_orbital_transformed[acc: acc + dim_cgfs] = fun(cgfs, xyz)
        acc += dim_cgfs

    return molecular_orbital_transformed


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

    # Accumlate x, y and z for each one of the primitves
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


def calculate_fourier_trasform_primitive(l: int, x: float, k: float,
                                         alpha: float) -> complex:
    """
    Compute the fourier transform for primitive Gaussian Type Orbitals.
    """
    pik = pi * k
    f = exp(-alpha * x ** 2 + complex(alpha * x, - pik) ** 2 / alpha)
    if l == 0:
        return sqrt(pi / alpha) * f
    elif l == 1:
        f = k * exp(-pik * complex(pik / alpha, 2 * x))
        return (pi / alpha) ** 1.5  * f
    elif l == 2:
        f = exp(-pik * complex(pik / alpha, 2 * x))
        return sqrt(pi / (alpha ** 5)) * (alpha / 2 - pik ** 2) * f
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)


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
