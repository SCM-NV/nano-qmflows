"""Module containing physical constants and `NamedTuple`s to store molecular orbitals, shell, etc.

Index
-----
.. currentmodule:: nanoqm.common
.. autosummary::
    DictConfig
    change_mol_units
    getmass
    number_spherical_functions_per_atom
    retrieve_hdf5_data
    is_data_in_hdf5
    store_arrays_in_hdf5

API
---
.. autoclass:: DictConfig
.. autofunction:: is_data_in_hdf5
.. autofunction:: retrieve_hdf5_data
.. autofunction:: number_spherical_functions_per_atom
.. autofunction:: store_arrays_in_hdf5

"""

from __future__ import annotations

__all__ = ['DictConfig', 'Matrix', 'Tensor3D', 'Vector',
           'change_mol_units', 'getmass', 'h2ev', 'hardness',
           'number_spherical_functions_per_atom', 'retrieve_hdf5_data',
           'is_data_in_hdf5', 'store_arrays_in_hdf5', 'UniqueSafeLoader',
           'valence_electrons', 'aux_fit']

import os
import json
from itertools import chain
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, NamedTuple, Tuple,
                    Sequence, overload, TypeVar, TYPE_CHECKING, Iterator)

import h5py
import mendeleev
import numpy as np
from scipy.constants import physical_constants
from qmflows.common import AtomXYZ
from qmflows.type_hints import PathLike
from scm.plams import Atom, Molecule

from qmflows.yaml_utils import UniqueSafeLoader
from . import __path__ as nanoqm_path

if TYPE_CHECKING:
    import numpy.typing as npt

_T = TypeVar("_T")


_path_valence_electrons = Path(nanoqm_path[0]) / "basis" / "valence_electrons.json"
_path_aux_fit = Path(nanoqm_path[0]) / "basis" / "aux_fit.json"

with open(_path_valence_electrons, 'r') as f1, open(_path_aux_fit, 'r') as f2:
    valence_electrons: "dict[str, int]" = json.load(f1)
    aux_fit: "dict[str, list[int]]" = json.load(f2)


class DictConfig(Dict[str, Any]):
    """Class to extend the Dict class with `.` dot notation."""

    def __getattr__(self, attr: str) -> Any:
        """Extract key using dot notation."""
        return self.get(attr)

    def __setattr__(self, key: str, value: Any) -> None:
        """Set value using dot notation."""
        self.__setitem__(key, value)

    def __deepcopy__(self, _: object) -> "DictConfig":
        """Deepcopy of the Settings object."""
        return DictConfig(self.copy())


class BasisFormats(NamedTuple):
    """NamedTuple that contains the name/value for the basis formats."""

    name: str
    value: list[npt.NDArray[np.int64]]


def concat(xss: Iterable[Iterable[_T]]) -> List[_T]:
    """Concatenate of all the elements of a list."""
    return list(chain(*xss))


# ================> Constants <================

# Prevent a TypeError whenever scipy is mocked
if isinstance(physical_constants['atomic unit of length'][0], float):
    #: Angstrom to a.u
    angs2au = 1e-10 / physical_constants['atomic unit of length'][0]
    #: from femtoseconds to au
    femtosec2au = 1e-15 / physical_constants['atomic unit of time'][0]
    #: hartrees to electronvolts
    h2ev = physical_constants['Hartree energy in eV'][0]
    #: conversion from rydberg to meV
    r2meV = 1e3 * physical_constants['Rydberg constant times hc in eV'][0]
    #: conversion from fs to cm-1
    fs_to_cm = 1e13 * physical_constants['hertz-inverse meter relationship'][0]
    #: conversion from fs to nm
    fs_to_nm = 299.79246
    #: planck constant in eV * fs
    hbar = 1e15 * physical_constants['Planck constant over 2 pi in eV s'][0]
else:
    angs2au = femtosec2au = h2ev = r2meV = fs_to_cm = fs_to_nm = hbar = 1.0

# type hints
MolXYZ = List[AtomXYZ]
Vector = np.ndarray
Matrix = np.ndarray
Tensor3D = np.ndarray


def getmass(s: str) -> int:
    """Get the atomic mass for a given element s."""
    element = mendeleev.element(s.capitalize())
    return element.mass_number


def hardness(s: str) -> float:
    """Get the element hardness."""
    d = {
        'h': 6.4299, 'he': 12.5449, 'li': 2.3746, 'be': 3.4968, 'b': 4.619, 'c': 5.7410,
        'n': 6.8624, 'o': 7.9854, 'f': 9.1065, 'ne': 10.2303, 'na': 2.4441, 'mg': 3.0146,
        'al': 3.5849, 'si': 4.1551, 'p': 4.7258, 's': 5.2960, 'cl': 5.8662, 'ar': 6.4366,
        'k': 2.3273, 'ca': 2.7587, 'sc': 2.8582, 'ti': 2.9578, 'v': 3.0573, 'cr': 3.1567,
        'mn': 3.2564, 'fe': 3.3559, 'co': 3.4556, 'ni': 3.555, 'cu': 3.6544, 'zn': 3.7542,
        'ga': 4.1855, 'ge': 4.6166, 'as': 5.0662, 'se': 5.4795, 'br': 5.9111, 'kr': 6.3418,
        'rb': 2.1204, 'sr': 2.5374, 'y': 2.6335, 'zr': 2.7297, 'nb': 2.8260, 'mo': 2.9221,
        'tc': 3.0184, 'ru': 3.1146, 'rh': 3.2107, 'pd': 3.3069, 'ag': 3.4032, 'cd': 3.4994,
        'in': 3.9164, 'sn': 4.3332, 'sb': 4.7501, 'te': 5.167, 'i': 5.5839, 'xe': 6.0009,
        'cs': 0.6829, 'ba': 0.9201, 'la': 1.1571, 'ce': 1.3943, 'pr': 1.6315, 'nd': 1.8686,
        'pm': 2.1056, 'sm': 2.3427, 'eu': 2.5798, 'gd': 2.8170, 'tb': 3.0540, 'dy': 3.2912,
        'ho': 3.5283, 'er': 3.7655, 'tm': 4.0026, 'yb': 4.2395, 'lu': 4.4766, 'hf': 4.7065,
        'ta': 4.9508, 'w': 5.1879, 're': 5.4256, 'os': 5.6619, 'ir': 5.900, 'pt': 6.1367,
        'au': 6.3741, 'hg': 6.6103, 'tl': 1.7043, 'pb': 1.9435, 'bi': 2.1785, 'po': 2.4158,
        'at': 2.6528, 'rn': 2.8899, 'fr': 0.9882, 'ra': 1.2819, 'ac': 1.3497, 'th': 1.4175,
        'pa': 1.9368, 'u': 2.2305, 'np': 2.5241, 'pu': 3.0436, 'am': 3.4169, 'cm': 3.4050,
        'bk': 3.9244, 'cf': 4.2181, 'es': 4.5116, 'fm': 4.8051, 'md': 5.0100, 'no': 5.3926,
        'lr': 5.4607}
    return d[s] / 27.211


def xc(s: str) -> Dict[str, Any]:
    """Return the exchange functional composition."""
    d = {
        'pbe': {
            'type': 'pure', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0, 'beta1': 0.2, 'beta2': 1.83},
        'blyp': {
            'type': 'pure', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0, 'beta1': 0.2, 'beta2': 1.83},
        'bp': {
            'type': 'pure', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0, 'beta1': 0.2, 'beta2': 1.83},
        'pbe0': {
            'type': 'hybrid', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.25, 'beta1': 0.2, 'beta2': 1.83},
        'b3lyp': {
            'type': 'hybrid', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.20, 'beta1': 0.2, 'beta2': 1.83},
        'bhlyp': {
            'type': 'hybrid', 'alpha1': 1.42, 'alpha2': 0.48, 'ax': 0.50, 'beta1': 0.2, 'beta2': 1.83},
        'cam-b3lyp': {
            'type': 'rhs', 'alpha1': 1.86, 'alpha2': 0.00, 'ax': 0.38, 'beta1': 0.90, 'beta2': 0},
        'lc-blyp': {
            'type': 'rhs', 'alpha1': 8.0, 'alpha2': 0.00, 'ax': 0.53, 'beta1': 4.50, 'beta2': 0},
        'wb97': {
            'type': 'rhs', 'alpha1': 8.0, 'alpha2': 0.00, 'ax': 0.61, 'beta1': 4.41, 'beta2': 0.0}}
    return d[s]


@overload
def retrieve_hdf5_data(path_hdf5: str | os.PathLike[str], paths_to_prop: str) -> npt.NDArray[Any]:
    ...


@overload
def retrieve_hdf5_data(path_hdf5: str | os.PathLike[str], paths_to_prop: List[str]) -> List[npt.NDArray[Any]]:
    ...


def retrieve_hdf5_data(
    path_hdf5: str | os.PathLike[str],
    paths_to_prop: str | list[str],
) -> npt.NDArray[Any] | list[npt.NDArray[Any]]:
    """Read Numerical properties from ``paths_hdf5``.

    Parameters
    ----------
    path_hdf5
        path to the HDF5
    path_to_prop
        str or list of str to data

    Returns
    -------
    np.ndarray
        array or list of array

    Raises
    ------
    RuntimeError
        The property has not been found

    """
    path_hdf5 = os.fspath(path_hdf5)
    try:
        with h5py.File(path_hdf5, 'r') as f5:
            if isinstance(paths_to_prop, list):
                return [f5[path][()] for path in paths_to_prop]
            else:
                return f5[paths_to_prop][()]
    except KeyError:
        msg = f"There is not {paths_to_prop} stored in the HDF5\n"
        raise KeyError(msg)
    except FileNotFoundError:
        msg = "there is not HDF5 file containing the numerical results"
        raise RuntimeError(msg)


def is_data_in_hdf5(path_hdf5: str | os.PathLike[str], xs: str | List[str]) -> bool:
    """Search if the node exists in the HDF5 file.

    Parameters
    ----------
    path_hdf5
        path to the HDF5
    xs
        either Node path or a list of paths to the stored data

    Returns
    -------
    bool
        Whether the data is stored

    """
    path_hdf5 = os.fspath(path_hdf5)
    if os.path.exists(path_hdf5):
        with h5py.File(path_hdf5, 'r+') as f5:
            if isinstance(xs, list):
                return all(path in f5 for path in xs)
            else:
                return xs in f5
    else:
        return False


def store_arrays_in_hdf5(
    path_hdf5: PathLike,
    paths: str | List[str],
    tensor: np.ndarray | Sequence[np.ndarray],
    dtype: npt.DTypeLike = np.float32,
    attribute: BasisFormats | None = None,
) -> None:
    """Store a tensor in the HDF5.

    Parameters
    ----------
    path_hdf5
        path to the HDF5
    paths
        str or list of nodes where the data is going to be stored
    tensor
        Numpy array or list of array to store
    dtype
        Data type use to store the numerical array
    attribute
        Attribute associated with the tensor

    """
    path_hdf5 = os.fspath(path_hdf5)

    def add_attribute(data_set, k: int = 0):
        if attribute is not None:
            data_set.attrs[attribute.name] = attribute.value[k]

    with h5py.File(path_hdf5, 'r+') as f5:
        if isinstance(paths, list):
            for k, path in enumerate(paths):
                data = tensor[k]
                dset = f5.require_dataset(path, shape=np.shape(data),
                                          data=data, dtype=dtype)
                add_attribute(dset, k)
        else:
            dset = f5.require_dataset(paths, shape=np.shape(
                tensor), data=tensor, dtype=dtype)
            add_attribute(dset)


def change_mol_units(mol: List[AtomXYZ], factor: float = angs2au) -> List[AtomXYZ]:
    """Change the units of the molecular coordinates."""
    new_molecule = []
    for atom in mol:
        coord = tuple(map(lambda x: x * factor, atom.xyz))
        new_molecule.append(AtomXYZ(atom.symbol, coord))  # type: ignore[arg-type]
    return new_molecule


def tuplesXYZ_to_plams(xs: List[AtomXYZ]) -> Molecule:
    """Transform a list of namedTuples to a Plams molecule."""
    plams_mol = Molecule()
    for at in xs:
        symb = at.symbol
        cs = at.xyz
        plams_mol.add_atom(Atom(symbol=symb, coords=tuple(cs)))

    return plams_mol


def number_spherical_functions_per_atom(
    mol: List[AtomXYZ],
    package_name: str,
    basis_name: str,
    path_hdf5: PathLike,
) -> npt.NDArray[np.int_]:
    """Compute the number of spherical shells per atom."""
    ret = []
    with h5py.File(path_hdf5, 'r') as f:
        iterator = ((at_tup[0], valence_electrons[at_tup[0].capitalize()]) for at_tup in mol)
        for atom, q in iterator:
            grp = f[f'{package_name}/basis/{atom}/{basis_name}-q{q}']
            n_funcs = sum(
                calc_n_spherics(grp[f"{i}/coefficients"].attrs['basisFormat'][4:]) for i in grp
            )
            ret.append(n_funcs)
    return np.fromiter(ret, dtype=int)


def calc_n_spherics(fss: npt.NDArray[np.int_]) -> np.int_:
    """Compute the number of spherical shells for a given basis set.

    Most quantum packages use standard basis set which contraction is
    presented usually by a format like:
    c def2-SV(P)
    # c     (7s4p1d) / [3s2p1d]     {511/31/1}
    this mean that this basis set for the Carbon atom uses 7 ``s`` CGF,
    4 ``p`` CGF and 1 ``d`` CGFs that are contracted in 3 groups of 5-1-1
    ``s`` functions, 3-1 ``p`` functions and 1 ``d`` function. Therefore
    the basis set format can be represented by [[5,1,1], [3,1], [1]].

    On the other hand Cp2k uses a special basis set ``MOLOPT`` which
    format explanation can be found at: `C2pk
    <https://github.com/cp2k/cp2k/blob/e392d1509d7623f3ebb6b451dab00d1dceb9a248/cp2k/data/BASIS_MOLOPT>`_.

    Parameters
    ----------
    fss : np.ndarray
        Format basis set

    Returns
    -------
    int
        The number of spherical shells.

    """
    return (fss * np.arange(1, 1 + 2 * len(fss), 2)).sum()


def read_cell_parameters_as_array(
    file_cell_parameters: str | os.PathLike[str],
) -> Tuple[str, npt.NDArray[np.float64]]:
    """Read the cell parameters as a numpy array."""
    arr = np.loadtxt(file_cell_parameters, skiprows=1)

    with open(file_cell_parameters, 'r') as f:
        header = f.readline()

    return header, arr
