__all__ = [
    'schema_cp2k_general_settings', 'schema_derivative_couplings',
    'schema_distribute_derivative_couplings',
    'schema_absorption_spectrum']


from numbers import Real
from schema import (And, Optional, Or, Schema, Use)
import os


def merge(d1, d2):
    """
    merge two dictionaries using without modifying the original
    """
    x = d1.copy()

    x.update(d2)

    return x


schema_cp2k_general_settings = Schema({

    # "Basis set to carry out the quantum chemistry simulation"
    "basis": str,

    # "Pseudo-potential to carry out the quantum chemistry simulation"
    "potential": str,

    # Specify the Cartesian components for the cell vector
    "cell_parameters": Or(
        Real,
        lambda xs: len(xs) == 3 and isinstance(xs, list),
        lambda xs: len(xs) == 3 and all(len(r) == 3 for r in xs)),

    # Type of periodicity
    "periodic":  And(
        str, Use(str.lower), lambda s: s in (
            "none", "x", "y", "z", "xy", "xy", "yz", "xyz")),

    # Specify the angles between the vectors defining the unit cell
    Optional("cell_angles", default=[90, 90, 90]): list,

    # Path to the folder containing the basis set specifications
    Optional("path_basis", default=None): str,

    # Path to the folder containing the pseudo potential specifications
    Optional("path_potential", default=None): str,

    # Settings describing the input of the quantum package
    "cp2k_settings_main": object,

    # Settings describing the input of the quantum package
    # to compute the guess wavefunction"
    "cp2k_settings_guess": object

    # # Restart File Name
    # Optional("wfn_restart_file_name", default=None): str,
})

dict_general_options = {

    # Number of occupied/virtual orbitals to use
    'active_space': [int, int],

    # Index of the HOMO
    Optional("nHOMO"): int,

    # "default quantum package used"
    Optional("package_name", default="cp2k"): str,

    # project
    Optional("project_name", default="namd"): str,

    # Working directory
    Optional("scratch_path", default=None): str,

    # path to the HDF5 to store the results
    Optional("path_hdf5", default="quantum.hdf5"): str,

    # path to xyz trajectory of the Molecular dynamics
    "path_traj_xyz": os.path.exists,

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

    # Integration time step used for the MD (femtoseconds)
    Optional("dt", default=1): Real,

    # General settings
    "cp2k_general_settings": schema_cp2k_general_settings
}

dict_derivative_couplings = {
    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "derivative_couplings"),

    # Algorithm used to compute the derivative couplings
    Optional("algorithm", default="levine"):
    And(str, Use(str.lower), lambda s: ("levine", "3points")),

    # Track the crossing between states
    Optional("tracking", default=True): bool,

    # Write the overlaps in ascii
    Optional("write_overlaps", default=False): bool,

    # Compute the overlap between molecular geometries using a dephase"
    Optional("overlaps_deph", default=False): bool
}

dict_merged_derivative_couplings = merge(dict_general_options, dict_derivative_couplings)

schema_derivative_couplings = Schema(
    dict_merged_derivative_couplings)

schema_job_scheduler = Schema({
    Optional("scheduler", default="SLURM"):
    And(str, Use(str.upper), lambda s: ("SLURM", "PBS")),
    Optional("nodes", default=1): int,
    Optional("tasks", default=1): int,
    Optional("wall_time", default="01:00:00"): str,
    Optional("job_name", default="namd"): str,
    Optional("load_modules", default=""): str
})

dict_distribute_derivative_couplings = {

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "distribute_derivative_couplings"),

    Optional("workdir", default=os.getcwd()): str,

    # Number of chunks to split the trajectory
    "blocks": int,

    # Resource manager configuration
    "job_scheduler": schema_job_scheduler,

    # General settings
    "cp2k_general_settings": schema_cp2k_general_settings,

}


schema_distribute_derivative_couplings = Schema(
    merge(dict_merged_derivative_couplings, dict_distribute_derivative_couplings))


dict_absorption_spectrum = {

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "absorption_spectrum"),

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
}


dict_merged_absorption_spectrum = merge(dict_general_options, dict_absorption_spectrum)

# define schema
schema_absorption_spectrum = Schema(dict_merged_absorption_spectrum)


dict_distribute_absorption_spectrum = {

    # Name of the workflow to run
    "workflow": And(
        str, Use(str.lower), lambda s: s == "distribute_absorption_spectrum"),

    Optional("workdir", default=os.getcwd()): str,

    # Number of chunks to split the trajectory
    "blocks": int,

    # Resource manager configuration
    "job_scheduler": schema_job_scheduler,

    # General settings
    "cp2k_general_settings": schema_cp2k_general_settings,

}

# define distribute schema
schema_distribute_absorption_spectrum = Schema(
    merge(dict_merged_absorption_spectrum, dict_distribute_absorption_spectrum))
