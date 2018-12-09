from nac.workflows.input_validation import process_input
from qmflows import (cp2k, run)
from scm import plams
import distutils
import pytest


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
    path_input = "test/test_files/input_test_templates.yml"
    dict_input = process_input(path_input, "absorption_spectrum")
    sett = dict_input['general_settings']['settings_guess']

    pbe = sett.specific.cp2k.force_eval.dft.xc.xc_functional

    assert("pbe" == pbe)


@pytest.mark.skipif(
    not cp2k_available(), reason="CP2K is not install or not loaded")
def test_call_cp2k():
    """
    Check if the input for a cp2k job is valid
    """
    path_input = "test/test_files/input_test_templates.yml"
    dict_input = process_input(path_input, "absorption_spectrum")
    sett = dict_input['general_settings']['settings_guess']
    job = cp2k(sett, plams.Molecule("test/test_files/Cd.xyz"))

    results = run(job.energy)

    print(results)
