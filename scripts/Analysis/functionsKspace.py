
from cmath import (exp, pi, sqrt)
from functools import partial

import h5py
import numpy as np


# Some Hint about the types
from typing import  List, NamedTuple
Vector = np.ndarray
Matrix = np.ndarray


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
