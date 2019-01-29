__all__ = [
    'schema_general_settings', 'schema_derivative_couplings',
    'schema_distribute_derivative_couplings',
    'schema_absorption_spectrum', 'schema_electron_transfer']


from numbers import Real
from schema import (And, Optional, Schema, Use)
import os

schema_general_settings = Schema({
    # "Library to distribute the computation"
    Optional("runner", default="multiprocessing"):
    And(str, Use(str.lower),
        lambda s: s in ("multiprocessing", "mpi")),

    # "default quantum package used"
    Optional("package_name", default="cp2k"): str,

    # project
    Optional("project_name", default="namd"): str,

    # "Basis set to carry out the quantum chemistry simulation"
    "basis_name": str,

    # Index of the HOMO
    "nHOMO": int,

    # Range of orbitals to print
    "mo_index_range": Schema([int, int]),

    # Working directory
    Optional("scratch_path", default="/tmp"): str,

    # path to the HDF5 to store the results
    "path_hdf5": str,

    # path to xyz trajectory of the Molecular dynamics
    "path_traj_xyz": os.path.exists,

    # Path to the folder containing the basis set specifications
    Optional("path_basis", default=None): str,

    # Path to the folder containing the pseudo potential specifications
    Optional("path_potential", default=None): str,

    # Real from where to start enumerating the folders create for each point in the MD
    Optional("enumerate_from", default=0): int,

    # Ignore the warning issues by the quantum package and keep computing
    Optional("ignore_warnings", default=False): bool,

    # Calculate the guess wave function in either the first point of the trajectory or in all
    Optional("calculate_guesses", default="first"):
    And(str, Use(str.lower), lambda s: s in ("first", "all")),

    # Units of the molecular geometry on the MD file
    Optional("geometry_units", default="angstrom"):
    And(str, Use(str.lower), lambda s: s in (
        "angstrom", "au")),

    # # Restart File Name
    # Optional("wfn_restart_file_name", default=None): str,

    # Settings describing the input of the quantum package
    "settings_main": object,

    # Settings describing the input of the quantum package
    # to compute the guess wavefunction"
    "settings_guess": object
})


schema_derivative_couplings = Schema({

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "derivative_couplings"),

    # Integration time step used for the MD (femtoseconds)
    Optional("dt", default=1): Real,

    # Algorithm used to compute the derivative couplings
    Optional("algorithm", default="levine"):
    And(str, Use(str.lower), lambda s: ("levine", "3points")),

    # Track the crossing between states
    Optional("tracking", default=True): bool,

    # Write the overlaps in ascii
    Optional("write_overlaps", default=False): bool,

    # Compute the overlap between molecular geometries using a dephase"
    Optional("overlaps_deph", default=False): bool,

    # General settings
    "general_settings": schema_general_settings
})

schema_job_scheduler = Schema({
    Optional("scheduler", default="SLURM"):
    And(str, Use(str.upper), lambda s: ("SLURM", "PBS")),
    Optional("nodes", default=1): int,
    Optional("tasks", default=1): int,
    Optional("wall_time", default="01:00:00"): str,
    Optional("job_name", default="namd"): str,
    Optional("load_modules", default=""): str
})


schema_distribute_derivative_couplings = Schema({

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "distribute_derivative_couplings"),

    # Integration time step used for the MD (femtoseconds)
    Optional("dt", default=1): Real,

    Optional("workdir", default=os.getcwd()): str,

    # Algorithm used to compute the derivative couplings
    Optional("algorithm", default="levine"):
    And(str, Use(str.lower), lambda s: ("levine", "3points")),

    # Track the crossing between states
    Optional("tracking", default=True): bool,

    # Write the overlaps in ascii
    Optional("write_overlaps", default=False): bool,

    # Compute the overlap between molecular geometries using a dephase"
    Optional("overlaps_deph", default=False): bool,

    # General settings
    "general_settings": schema_general_settings,

    # Number of chunks to split the trajectory
    "blocks": int,

    # Resource manager configuration
    "job_scheduler": schema_job_scheduler
})


schema_absorption_spectrum = Schema({

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "absorption_spectrum"),

    # Initial states of the transitions
    Optional("initial_states"): list,

    # final states of the transitions (Array or Arrays)
    Optional("final_states"): list,

    # Type of TDDFT calculations. Available: sing_orb, stda, stddft
    Optional("tddft", default="stda"): And(
        str, Use(str.lower), lambda s: s in ("sing_orb", "stda", "stdft")),

    # Range of energy in eV to simulate the spectrum"
    Optional("energy_range", default=[0, 5]): Schema([Real, Real]),

    # Interval between MD points where the oscillators are computed"
    Optional("calculate_oscillator_every",  default=50): int,

    # description: Exchange-correlation functional used in the DFT calculations,
    Optional("xc_dft", default="pbe"): str,

    # mathematical function representing the spectrum,
    Optional("convolution", default="gaussian"): And(
        str, Use(str.lower), lambda s: s in ("gaussian", "lorentzian")),

    # thermal broadening in eV
    Optional("broadening", default=0.1): Real,

    # General settings
    "general_settings": schema_general_settings
})


schema_electron_transfer = Schema({
    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "electron_transfer"),

    # Path to the PYXAID output containing the time-dependent coefficients
    "path_time_coeffs": str,

    # Integration time step used for the MD (femtoseconds)
    Optional("dt", default=1): float,

    # Index of the HOMO
    "pyxaid_HOMO": int,

    # Index of the LUMO
    "pyxaid_LUMO": int,

    # Index of the LUMO
    "pyxaid_Nmax": int,

    # List of initial conditions of the Pyxaid dynamics
    "pyxaid_iconds": list,

    # Indices of the atoms belonging to a fragment
    "fragment_indices": list,

    # Range of Molecular orbitals use to compute the derivative couplings
    "mo_index_range": list,

    # General settings
    "general_settings": schema_general_settings
})
