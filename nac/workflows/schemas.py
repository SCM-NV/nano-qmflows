__all__ = ['schema_general_settings', 'schema_derivative_couplings']


from schema import (And, Optional, Or, Schema, Use)


schema_general_settings = Schema({
    # "Library to distribute the computation"
    Optional("runner", default="multiprocessing"):
    And(str, Use(str.lower),
        lambda s: s in ("multiprocessing", "mpi")),

    # "default quantum package used"
    Optional("project_name", default="namd"): str,

    # "Basis set to carry out the quantum chemistry simulation"
    "basis_name": str,

    # Working directory
    Optional("scratch_path", default="/tmp"): str,

    # path to the HDF5 to store the results
    "path_hdf5": str,

    # path to xyz trajectory of the Molecular dynamics
    "path_traj_xyz": str,

    # Path to the folder containing the basis set specifications
    Optional("path_basis", default=None): str,

    # Path to the folder containing the pseudo potential specifications
    Optional("path_potential", default=None): str,

    # Number from where to start enumerating the folders create for each point in the MD
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

    # Settings describing the input of the quantum package
    "settings_main": object,

    # Settings describing the input of the quantum package
    # to compute the guess wavefunction"
    "settings_guess": object
})


definitions_derivative_couplings = Schema({

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "derivative_coupling"),

    # Index of the HOMO
    "nHOMO": int,

    # Integration time step used for the MD (femtoseconds)
    Optional("dt", default=1): float,

    # Range of Molecular orbitals used to compute the nonadiabatic coupling matrix
    "couplings_range": Schema((int, int)),

    # Algorithm used to compute the derivative couplings
    Optional("algorithm", default="levine"):
    And(str, Use(str.lower), lambda s: ("levine", "3points")),

    # Track the crossing between states
    Optional("tracking", default=True): bool,

    # Write the overlaps in ascii
    Optional("write_overlaps", default=False): bool,

    # Compute the overlap between molecular geometries using a dephase"
    Optional("overlaps_deph", default=False): bool

})


definitions_absorption_spectrum = Schema({
    
})


schema_derivative_couplings = Schema(
    And(schema_general_settings, definitions_derivative_couplings))
