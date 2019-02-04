from nac.workflows.input_validation import process_input
from qmflows import (cp2k, run)
from scm import plams
import distutils
import pytest
import fnmatch
import shutil
import os


def cp2k_available():
    """
    Check if cp2k is installed
    """
    path = distutils.spawn.find_executable("cp2k.popt")

    return path is not None


def test_input_validation():
    """
    test the templates and keywords completion
    """
    path_input = "test/test_files/input_test_pbe0.yml"
    dict_input = process_input(path_input, "derivative_couplings")
    sett = dict_input['cp2k_general_settings']['cp2k_settings_guess']

    print("sett: ", sett)

    scale_x = sett.specific.cp2k.force_eval.dft.xc.xc_functional.pbe.scale_x

    assert abs(scale_x - 0.75) < 1e-16


# @pytest.mark.skipif(
#     not cp2k_available(), reason="CP2K is not install or not loaded")
# def test_call_cp2k():
#     """
#     Check if the input for a cp2k job is valid
#     """
#     try:
#         path_input = "test/test_files/input_test_pbe.yml"
#         dict_input = process_input(path_input, "absorption_spectrum")
#         sett = dict_input['general_settings']['cp2k_settings_guess']
#         job = cp2k(sett, plams.Molecule("test/test_files/Cd.xyz"))

#         results = run(job.energy)

#         print("sett: ", sett)

#         assert (results is not None)

#     finally:
#         for path in fnmatch.filter(os.listdir('.'), "plams_workdir"):
#             shutil.rmtree(path)
#         if os.path.exists("cache.db"):
#             os.remove("cache.db")
