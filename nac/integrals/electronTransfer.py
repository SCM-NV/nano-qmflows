__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from nac.common import (Matrix, Tensor3D)
from typing import (List, Tuple)
import numpy as np


def photo_excitation_rate(
        geometries: Tuple, tensor_overlaps: Tensor3D,
        time_dependent_coeffs: Matrix, map_index_pyxaid_hdf5: Matrix,
        dt_au: float) -> Tuple:
    """
    Calculate the Electron transfer rate, using both adiabatic and nonadiabatic
    components, using equation number 8 from:
    J. AM. CHEM. SOC. 2005, 127, 7941-7951. Ab Initio Nonadiabatic Molecular
    Dynamics of the Ultrafast Electron Injection acrossthe Alizarin-TiO2
    Interface.
    The derivatives are calculated numerically using 3 points.

    :param geometry: Molecular geometries.
    :type geometry: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :param tensor_overlaps: Overlap matrices at time t - dt, t and t + dt.
    :param time_dependent_coeffs: Time-dependentent coefficients
    at time t - dt, t and t + dt.
    :param map_index_pyxaid_hdf5: Index relation between the Excitations
    in PYXAID and the corresponding molecular orbitals store in the HDF5.
    :param dt_au: Delta time integration used in the dynamics.
    :returns: tuple containing both nonadiabatic and adiabatic components
    """
    # indices of the i -> j transitions used by PYXAID
    row_indices = map_index_pyxaid_hdf5[:, 0]
    col_indices = map_index_pyxaid_hdf5[:, 1]

    # Rearrange the overlap matrix in the PYXAID order
    matrix_overlap_pyxaid_order = tensor_overlaps[1][row_indices, col_indices]

    # NonAdiabatic component
    coeff_derivatives = np.apply_along_axis(
        lambda v: (v[0] - v[2]) / (2 * dt_au), 0, time_dependent_coeffs)

    nonadiabatic = np.sum(coeff_derivatives * matrix_overlap_pyxaid_order)

    # Adiabatic component
    overlap_derv = np.apply_along_axis(
        lambda v: (v[0] - v[2]) / (2 * dt_au), 0, tensor_overlaps)

    overlap_derv_pyxaid_order = overlap_derv[row_indices, col_indices]

    adiabatic = np.sum(time_dependent_coeffs[1] * overlap_derv_pyxaid_order)

    return nonadiabatic, adiabatic
