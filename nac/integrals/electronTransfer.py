__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from nac.common import (Matrix, Tensor3D)
from typing import (List, Tuple)
import numpy as np


def photo_excitation_rate(
        geometries: Tuple, time_dependent_coeffs: Tensor3D,
        tensor_overlaps: Tensor3D, dt_au: float) -> Tuple:
    """
    Calculate the Electron transfer rate, using both adiabatic and nonadiabatic
    components, using equation number 8 from:
    J. AM. CHEM. SOC. 2005, 127, 7941-7951. Ab Initio Nonadiabatic Molecular
    Dynamics of the Ultrafast Electron Injection acrossthe Alizarin-TiO2
    Interface.
    The derivatives are calculated numerically using 3 points.

    :param geometry: Molecular geometries.
    :type geometry: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :param time_dependent_coeffs: Time-dependentent coefficients
    at time t - dt, t and t + dt.
    :param tensor_overlaps: Overlap matrices at time t - dt, t and t + dt.
    :param dt_au: Delta time integration used in the dynamics.
    :returns: tuple containing both nonadiabatic and adiabatic components
    """
    # NonAdiabatic component
    nonadiabatic = electronTransferNA(
        geometries[1], time_dependent_coeffs, tensor_overlaps[1], dt_au)
    # Adiabatic component
    css1 = time_dependent_coeffs[1]
    adiabatic = electronTransferAdiabatic(css1, tensor_overlaps)

    return nonadiabatic, adiabatic


def electronTransferNA(
        geometry: List, time_dependent_coeffs: Tensor3D, overlap: Matrix, dt: float):
    """
    Calculate the Nonadiabatic component of an Electron transfer process.

    :param geometry: Molecular geometry
    :param _coefficients: Time-dependentent coefficients at time t - dt, t and t + dt.
    :param overlap: Overlap matrix
    :param dt: Integration time.
    :returns: float
    """
    # Use a second order derivative for the time dependentent coefficients
    coeff_derivatives = np.apply_along_axis(
        lambda v: (v[0] - v[2]) / (2 * dt), 0, time_dependent_coeffs)

    return np.sum(coeff_derivatives * overlap)


def electronTransferAdiabatic(
        coefficients: Matrix, tensor_overlap: Tensor3D, dt: float) -> float:
    """
    Calculate the Adiabatic component of an Electron transfer process.

    :param coefficients: Time-dependent coefficients at time t.
    :param tensor_overlap: Tensor containing the overlap matrices at time
    t - dt, t and t + dt.
    returns: float
    """
    overlap_derv = np.apply_along_axis(
        lambda v: (v[0] - v[2]) / (2 * dt), 0, tensor_overlap)

    return np.sum(coefficients * overlap_derv)
