__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from noodles import schedule  # Workflow Engine
from os.path import join

import fnmatch
import os

# ==================> Internal modules <==========
from qmflows import templates
from qmflows.packages import cp2k
from qmflows.parsers.xyzParser import string_to_plams_Molecule

from typing import (Dict, Tuple)
# ==============================> Schedule Tasks <=========================


def prepare_cp2k_settings(files: Tuple, cp2k_args: Dict, k: int,
                          wfn_restart_job, cp2k_config: Dict) -> Dict:
    """
    Fills in the parameters for running a single job in CP2K.

    :param files: Tuple containing the IO files to run the calculations
    :parameter dict_input: Dictionary contaning the data to
    fill in the template
    :param k: nth Job
    :param wfn_restart_job: Path to *.wfn cp2k file use as restart file.
    :param cp2k_config:  Parameters required by cp2k.
   :returns: ~qmflows.Settings
    """
    # Search for the environmental variable BASISCP2K containing the path
    # to the Basis set folder
    basis_file = cp2k_config["basis"]
    potential_file = cp2k_config["potential"]

    dft = cp2k_args.specific.cp2k.force_eval.dft
    dft.basis_set_file_name = basis_file
    dft.potential_file_name = potential_file
    dft['print']['mo']['filename'] = files.get_MO

    # Global parameters for CP2K
    cp2k_args.specific.cp2k['global']['project'] = 'point_{}'.format(k)
    cp2k_args.specific.cp2k['global']['run_type'] = 'Energy'

    if wfn_restart_job is not None:
        output_dir = getattr(wfn_restart_job.archive['plams_dir'], 'path')
        xs = os.listdir(output_dir)
        wfn_file = list(filter(lambda x: fnmatch.fnmatch(x, '*wfn'), xs))[0]
        file_path = join(output_dir, wfn_file)
        dft.wfn_restart_file_name = file_path

    input_args = templates.singlepoint.overlay(cp2k_args)

    # Do not print the MOs if is an OT computation
    if cp2k_args.specific.cp2k.force_eval.dft.scf.ot:
        del input_args.specific.cp2k.force_eval.dft['print']['mo']

    return input_args


@schedule
def prepare_job_cp2k(geometry: str, files: Tuple, dict_input: Dict, k: int,
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
    job_settings = prepare_cp2k_settings(files, dict_input, k,
                                         wfn_restart_job, package_config)

    return cp2k(job_settings, string_to_plams_Molecule(geometry), work_dir=work_dir)
