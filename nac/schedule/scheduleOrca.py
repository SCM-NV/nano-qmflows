__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from noodles import schedule  # Workflow Engine
from qmflows.packages import orca
from qmflows.parsers.xyzParser import string_to_plams_Molecule

from typing import (Dict, Tuple)
# ==============================> Schedule Tasks <=========================


@schedule
def prepare_job_orca(geometry: str, files: Tuple, settings: Dict, k: int,
                     work_dir: str, wfn_restart_job=None,
                     package_config: Dict=None):
    """
    Fills in the parameters for running a single job in CP2K.

    :param geometry: Molecular geometry stored as String
    :param files: Tuple containing the IO files to run the calculations
    :parameter dict_input: Dictionary contaning the data to
    fill in the template
    :param k: nth Job
    :parameter work_dir: Name of the Working folder
    :param wfn_restart_job: Path to *.wfn cp2k file use as restart file.
    :returns: ~qmflows.CP2K
    """

    return orca(settings, string_to_plams_Molecule(geometry))
