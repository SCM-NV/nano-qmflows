# ================> Python Standard  and third-party <==========
from noodles import schedule  # Workflow Engine
from os.path import join

import fnmatch
import os

# ==================> Internal modules <==========
from qmflows import templates
from qmflows.packages import cp2k
from qmflows.parsers.xyzParser import string_to_plams_Molecule


def prepare_cp2k_settings(settings: object, dict_input: dict, guess_job: object) -> object:
    """
    Fills in the parameters for running a single job in CP2K.

    :param files: Tuple containing the IO files to run the calculations
    :parameter dict_input: Dictionary contaning the data to
    fill in the template
    :param k: nth Job
    :param guess_job: Path to *.wfn cp2k file use as restart file.
    :param cp2k_config:  Parameters required by cp2k.
   :returns: ~qmflows.Settings
    """
    dft = settings.specific.cp2k.force_eval.dft
    dft['print']['mo']['filename'] = dict_input["job_files"].get_MO

    # Global parameters for CP2K
    settings.specific.cp2k['global']['project'] = 'point_{}'.format(dict_input["k"])
    settings.specific.cp2k['global']['run_type'] = 'Energy'

    if guess_job is not None:
        output_dir = guess_job.archive['plams_dir']
        xs = os.listdir(output_dir)
        wfn_file = list(filter(lambda x: fnmatch.fnmatch(x, '*wfn'), xs))[0]
        file_path = join(output_dir, wfn_file)
        dft.wfn_restart_file_name = file_path

    input_args = templates.singlepoint.overlay(settings)

    # Do not print the MOs if is an OT computation
    if settings.specific.cp2k.force_eval.dft.scf.ot:
        del input_args.specific.cp2k.force_eval.dft['print']['mo']

    return input_args


@schedule
def prepare_job_cp2k(settings: object, dict_input: dict, guess_job: object) -> object:
    """
    Fills in the parameters for running a single job in CP2K.

    :param settings: Settings to run cp2k
    :parameter dict_input: Dictionary contaning the data to complete the settings

    The `dict_input` contains:

    :param geometry: Molecular geometry stored as String
    :param files: Tuple containing the IO files to run the calculations
    :param k: nth Job
    :parameter workdir: Name of the Working folder
    :param guess_job: Path to *.wfn cp2k file use as restart file.
    :returns: ~qmflows.CP2K
    """
    job_settings = prepare_cp2k_settings(settings, dict_input, guess_job)

    return cp2k(
        job_settings, string_to_plams_Molecule(dict_input["geometry"]),
        work_dir=dict_input['point_dir'])
