__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import partial
from itertools import starmap
from nac.common import (Matrix, Tensor3D, femtosec2au)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from scipy import sparse
from typing import (Dict, List, Tuple)

import numpy as np

# ==================================<>=========================================


def photoExcitationRate(
        geometries: Tuple, dictCGFs: Dict, time_depend_coeffs: Matrix,
        mos_coefficients: Tensor3D, trans_mtx: Matrix, dt: float) -> Tuple:
    """
    Calculate the Electron transfer rate, using both adiabatic and nonadiabatic
    components, using equation number 8 from:
    J. AM. CHEM. SOC. 2005, 127, 7941-7951. Ab Initio Nonadiabatic Molecular
    Dynamics of the Ultrafast Electron Injection acrossthe Alizarin-TiO2
    Interface.
    The derivatives are calculated numerically using 3 points.

    :param geometry: Molecular geometries.
    :type geometry: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],AngularMomentum),
    Primitive = (Coefficient, Exponent)
    :param tuple_time_coefficients: Time-dependent coefficients
    at time t - dt, t and t + dt.
    :param tuple_mos_coefficients: Tuple of Molecular orbitals at
    time t - dt, t and t + dt.
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :returns: tuple containing both nonadiabatic and adiabatic components
    """
    # transform time to atomic units
    dt_au = dt * femtosec2au
    # Geometry at time t
    r1 = geometries[1]

    # Overlap matrices
    overlaps = np.stack(starmap(partial(overlap_molecular, dictCGFs, trans_mtx),
                                zip(geometries, time_depend_coeffs)))
    # NonAdiabatic component
    nonadiabatic = electronTransferNA(r1, time_depend_coeffs, overlaps[1], dt_au)
    # Adiabatic component
    css1 = time_depend_coeffs[1]
    adiabatic = electronTransferAdiabatic(css1, overlaps)

    return nonadiabatic, adiabatic


def electronTransferNA(
        geometry: List, time_depend_coeffs: Tensor3D, overlap: Matrix, dt: float):
    """
    Calculate the Nonadiabatic component of an Electron transfer process.

    :param geometry: Molecular geometry
    :type geometry: List of Atom objects
    :param tuple_coefficients: Time-dependent coefficients at time
    t - dt, t and t + dt.
    :type tuple_coefficients: Tuple of Numpy Arrays
    :param overlap: Overlap matrix
    :type overlap: Numpy Martrix
    :param dt: Integration time.
    """
    dim = tuple_coefficients[0].shape
    xs = [x.reshape(1, dim) for x in tuple_coefficients]
    xss = np.stack([x * np.transpose(x) for x in xs])
    # Use a second order derivative for the time dependent coefficients
    derivatives = np.apply_along_axis(threePointDerivative, 0, xss)

    return np.sum(derivatives * overlap)


def electronTransferAdiabatic(coefficients, tensor_overlap):
    """
    Calculate the Adiabatic component of an Electron transfer process.
    :param tuple_coefficients: Time-dependent coefficients at time t.
    :type tuple_coefficients: Numpy Array
    :param tensor_overlap: Tensor containing the overlap matrices at time
    t - dt, t and t + dt.
    :type tensor_overlap: Numpy 3-D array

    """
    dim = coefficients.shape
    coefficients = coefficients.reshape(1, dim)
    css = coefficients * np.transpose(coefficients)

    overlap_derv = np.apply_along_axis(threePointDerivative, 0, tensor_overlap)

    return np.sum(css * overlap_derv)


def overlap_molecular(
        dictCGFs: Dict, trans_mtx: Matrix, geometry: List, mos: Matrix) -> Matrix:
    """
    Calculate the Overlap matrix using the given geometry and Molecular
    orbitals.

    :param geometry: Molecular geometry
    :type geometry: List of Atom objects
    :param cgfsN: List of Contracted Gaussian functions
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy matrix
    :param mos: Molecular orbitals coefficients
    :type mos: Numpy Martrix
    """
    atomic_overlap = calcMtxMultipoleP(geometry, dictCGFs)
    mosT = np.transpose(mos)

    transpose = trans_mtx.transpose()
    # Overlap in Sphericals
    suv = trans_mtx.dot(sparse.csr_matrix.dot(atomic_overlap, transpose))

    return np.dot(mosT, np.dot(suv, mos))


def threePointDerivative(arr, dt):
    """Calculate the numerical derivatives using a Two points formula"""

    return (arr[0] - arr[2]) / (2 * dt)
