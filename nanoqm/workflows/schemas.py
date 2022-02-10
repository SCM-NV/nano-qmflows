"""Schemas to valid user input.

Index
-----
.. currentmodule:: nanoqm.workflows.schemas
.. autosummary::
    {autosummary}

API
---
{autodata}

"""
__all__ = [
    'schema_cp2k_general_settings',
    'schema_derivative_couplings',
    'schema_single_points',
    'schema_distribute_absorption_spectrum',
    'schema_distribute_derivative_couplings',
    'schema_distribute_single_points',
    'schema_absorption_spectrum',
    'schema_ipr',
    'schema_coop']

import os
from numbers import Real
from typing import Any, Dict, Iterable

import pkg_resources as pkg
from schema import And, Optional, Or, Regex, Schema, Use


def equal_lambda(name: str) -> And:
    """Create an schema checking that the keyword matches the expected value."""
    return And(
        str, Use(str.lower), lambda s: s == name)


def any_lambda(array: Iterable[str]) -> And:
    """Create an schema checking that the keyword matches one of the expected values."""
    return And(
        str, Use(str.lower), lambda s: s in array)


def merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries using without modifying the original."""
    x = d1.copy()

    x.update(d2)

    return x


def _parse_filenames(f: "None | str | list[str]") -> "None | list[str]":
    """Helper function for sanitizing one or multiple filenames."""
    if isinstance(f, str):
        return [f]
    elif f is None or (isinstance(f, list) and all(isinstance(i, str) for i in f)):
        return f
    else:
        raise TypeError(type(f).__name__)


#: Schema to validate the CP2K general settings
schema_cp2k_general_settings = Schema({

    # "Basis set to carry out the quantum chemistry simulation"
    "basis": str,

    # "Pseudo-potential to carry out the quantum chemistry simulation"
    "potential": str,

    # Charge of the system
    Optional("charge", default=0): int,

    # Multiplicity
    Optional("multiplicity", default=1): int,

    # Specify the Cartesian components for the cell vector Units Angstrom
    Optional("cell_parameters", default=10): Or(
        Real,
        lambda xs: len(xs) == 3 and isinstance(xs, list),
        lambda xs: len(xs) == 3 and all(len(r) == 3 for r in xs)),

    # Type of periodicity
    "periodic": any_lambda(("none", "x", "y", "z", "xy", "xy", "yz", "xyz")),

    # Specify the angles between the vectors defining the unit cell
    Optional("cell_angles"): list,

    # Path to the folder containing the basis set specifications
    Optional("path_basis", default=pkg.resource_filename("nanoqm", "basis")): os.path.isdir,

    # Name(s) of the basis set file(s) stored in ``path_basis``
    Optional("basis_file_name", default=None): Use(_parse_filenames),

    # Name(s) of the potential file(s) stored in ``path_basis``
    Optional("potential_file_name", default=None): Use(_parse_filenames),

    # Name(s) of the exchange part of the DFT functional`
    Optional("functional_x", default=None): str,

    # Name(s) of the correlation part of the DFT functional`
    Optional("functional_c", default=None): str,

    # Settings describing the input of the quantum package
    "cp2k_settings_main": object,

    # Settings describing the input of the quantum package
    # to compute the guess wavefunction"
    "cp2k_settings_guess": object,

    # Restart File Name
    Optional("wfn_restart_file_name", default=None): Or(str, None),

    # File containing the Parameters of the cell if those
    # parameters change during the MD simulation.
    Optional("file_cell_parameters", default=None): Or(str, None),

    # Quality of the auxiliar basis cFIT
    Optional("aux_fit", default="verygood"):
        any_lambda(("low", "medium", "good", "verygood", "excellent")),

    # executable name
    # "sdbg" Serial single core testing and debugging
    # "sopt" Serial general single core usage
    # "ssmp" Parallel (only OpenMP), single node, multi core
    # "pdbg" Parallel (only MPI) multi-node testing and debugging
    # "popt" Parallel (only MPI) general usage, no threads
    # "psmp" parallel (MPI + OpenMP) general usage, threading might improve scalability and memory usage

    Optional("executable", default="cp2k.psmp"):
        Regex(r'.*cp2k\.(?:popt|psmp|sdbg|sopt|ssmp|pdbg)', flags=2)  # flag 2 == IGNORECASE
})


#: Dictionary with the options common to all workflows
dict_general_options = {

    # Number of occupied/virtual orbitals to use
    Optional('active_space', default=[10, 10]): And(list, lambda xs: len(xs) == 2),

    # Index of the HOMO
    Optional("nHOMO"): int,

    # Index of the orbitals to compute the couplings
    Optional("mo_index_range"): tuple,

    # "default quantum package used"
    Optional("package_name", default="cp2k"): str,

    # project
    Optional("project_name", default="namd"): str,

    # Working directory
    Optional("scratch_path", default=None): Or(None, str),

    # path to the HDF5 to store the results
    Optional("path_hdf5", default="quantum.hdf5"): str,

    # path to xyz trajectory of the Molecular dynamics
    "path_traj_xyz": os.path.exists,

    # Real from where to start enumerating the folders create for each point
    # in the MD
    Optional("enumerate_from", default=0): int,

    # Ignore the warning issues by the quantum package and keep computing
    Optional("ignore_warnings", default=False): bool,

    # Calculate the guess wave function in either the first point of the
    # trajectory or in all
    Optional("calculate_guesses", default="first"):
    any_lambda(("first", "all")),

    # Units of the molecular geometry on the MD file
    Optional("geometry_units", default="angstrom"):
    any_lambda(("angstrom", "au")),

    # Integration time step used for the MD (femtoseconds)
    Optional("dt", default=1): Real,

    # Deactivate the computation of the orbitals for debugging purposes
    Optional("compute_orbitals", default=True): bool,

    # Flag to remove the log containing the orbitals for debugging purposes
    Optional("remove_log_file", default=False): bool,

    # General settings
    "cp2k_general_settings": schema_cp2k_general_settings,

    # Empty string for restricted calculation or either alpha/beta
    # for unrestricted calculation
    Optional("orbitals_type", default=""): any_lambda(("", "alphas", "betas", "both"))
}

#: Dict with input options to run a derivate coupling workflow
dict_derivative_couplings = {
    # Name of the workflow to run
    "workflow": equal_lambda("derivative_couplings"),

    # Algorithm used to compute the derivative couplings
    Optional("algorithm", default="levine"):
    any_lambda(("levine", "3points")),

    # Use MPI to compute the couplings
    Optional("mpi", default=False): bool,

    # Track the crossing between states
    Optional("tracking", default=True): bool,

    # Write the overlaps in ascii
    Optional("write_overlaps", default=False): bool,

    # Compute the overlap between molecular geometries using a dephase"
    Optional("overlaps_deph", default=False): bool
}

dict_merged_derivative_couplings = merge(
    dict_general_options, dict_derivative_couplings)

#: Schema to validate the input for a derivative coupling calculation
schema_derivative_couplings = Schema(
    dict_merged_derivative_couplings)

#: Schema to validate the input for a job scheduler
schema_job_scheduler = Schema({
    Optional("scheduler", default="slurm"):
    any_lambda(("slurm", "pbs")),
    Optional("nodes", default=1): int,
    Optional("tasks", default=1): int,
    Optional("wall_time", default="01:00:00"): str,
    Optional("job_name", default="namd"): str,
    Optional("queue_name", default="short"): str,
    Optional("load_modules", default=""): str,
    Optional("free_format", default=""): str
})

#: Input options to distribute a job
dict_distribute = {

    Optional("workdir", default=os.getcwd()): str,

    # Number of chunks to split the trajectory
    "blocks": int,

    # Resource manager configuration
    "job_scheduler": schema_job_scheduler,

    # General settings
    "cp2k_general_settings": schema_cp2k_general_settings,


}

#: input to distribute a derivative coupling job
dict_distribute_derivative_couplings = {

    # Name of the workflow to run
    "workflow": equal_lambda("distribute_derivative_couplings")
}


#: Schema to validate the input to distribute a derivate coupling calculation
schema_distribute_derivative_couplings = Schema(
    merge(
        dict_distribute,
        merge(
            dict_merged_derivative_couplings,
            dict_distribute_derivative_couplings)))

#: Input for an absorption spectrum calculation
dict_absorption_spectrum = {

    # Name of the workflow to run
    "workflow": equal_lambda("absorption_spectrum"),

    # Type of TDDFT calculations. Available: sing_orb, stda, stddft
    Optional("tddft", default="stda"): And(
        str, Use(str.lower), lambda s: s in ("sing_orb", "stda", "stdft")),

    # Interval between MD points where the oscillators are computed"
    Optional("stride", default=1): int,

    # description: Exchange-correlation functional used in the DFT
    # calculations,
    Optional("xc_dft", default="pbe"): str
}


dict_merged_absorption_spectrum = merge(
    dict_general_options, dict_absorption_spectrum)

#: Schema to validate the input for an absorption spectrum calculation
schema_absorption_spectrum = Schema(dict_merged_absorption_spectrum)


dict_distribute_absorption_spectrum = {

    # Name of the workflow to run
    "workflow": equal_lambda("distribute_absorption_spectrum")
}

schema_distribute_absorption_spectrum = Schema(
    merge(dict_distribute, merge(
        dict_merged_absorption_spectrum, dict_distribute_absorption_spectrum)))

dict_single_points = {
    # Name of the workflow to run
    "workflow": any_lambda(("single_points", "ipr_calculation", "coop_calculation")),

    # General settings
    "cp2k_general_settings": schema_cp2k_general_settings
}

#: input to distribute single point calculations
dict_distribute_single_points = {

    # Name of the workflow to run
    "workflow": equal_lambda("distribute_single_points")
}

#: Input for a Crystal Orbital Overlap Population calculation
dict_coop = {
    # List of the two elements to calculate the COOP for
    "coop_elements": list}


dict_merged_single_points = merge(dict_general_options, dict_single_points)

#: Schema to validate the input of a single pointe calculation
schema_single_points = Schema(dict_merged_single_points)

#: Schema to validate the input for a Inverse Participation Ratio calculation
schema_ipr = schema_single_points

#: Input for a Crystal Orbital Overlap Population calculation
dict_merged_coop = merge(dict_merged_single_points, dict_coop)

#: Schema to validate the input for a Crystal Orbital Overlap Population calculation
schema_coop = Schema(dict_merged_coop)

#: Schema to validate the input to distribute a single point calculation
schema_distribute_single_points = Schema(
    merge(dict_distribute, merge(
        dict_merged_single_points, dict_distribute_single_points)))
