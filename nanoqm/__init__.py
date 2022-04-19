"""Nano-QMFlows is a generic python library for computing (numerically) electronic properties \
for nanomaterials like the non-adiabatic coupling vectors (NACV) using several quantum \
chemical (QM) packages."""

from ._version import __version__ as __version__
from ._version_info import version_info as version_info
from ._logger import logger as logger

from .analysis import (
    autocorrelate, dephasing, convolute, func_conv, gauss_function,
    parse_list_of_lists, read_couplings, read_energies,
    read_energies_pyxaid, read_pops_pyxaid, spectral_density
)

from .integrals import (
    calculate_couplings_levine, compute_overlaps_for_coupling)

from .schedule import (calculate_mos, lazy_couplings)

from .workflows import (
    workflow_derivative_couplings, workflow_stddft)

__all__ = [
    'autocorrelate', 'calculate_couplings_levine', 'calculate_mos',
    'compute_overlaps_for_coupling', 'convolute', 'dephasing',
    'func_conv', 'gauss_function', 'lazy_couplings',
    'parse_list_of_lists', 'read_couplings', 'read_energies',
    'read_energies_pyxaid', 'read_pops_pyxaid', 'spectral_density',
    'workflow_derivative_couplings', 'workflow_stddft']
