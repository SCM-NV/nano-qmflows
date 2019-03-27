from .utilsTest import (cp2k_available, remove_files)
from nac.workflows.input_validation import process_input
from qmflows import (cp2k, run)
from scm import plams
import pytest


def test_input_validation():
    """
    test the templates and keywords completion
    """
    path_input = "test/test_files/input_test_pbe0.yml"
    dict_input = process_input(path_input, "derivative_couplings")
    sett = dict_input['cp2k_general_settings']['cp2k_settings_guess']

    scale_x = sett.specific.cp2k.force_eval.dft.xc.xc_functional.pbe.scale_x

    assert abs(scale_x - 0.75) < 1e-16


@pytest.mark.skipif(
    not cp2k_available(), reason="CP2K is not install or not loaded")
def test_call_cp2k_pbe():
    """
    Check if the input for a PBE cp2k job is valid
    """
    try:
        results = run_plams("test/test_files/input_test_pbe.yml")
        assert (results is not None)
    finally:
        remove_files()


def run_plams(path_input):
    """
    Call Plams to run a CP2K job
    """
    dict_input = process_input(path_input, "derivative_couplings")
    sett = dict_input['cp2k_general_settings']['cp2k_settings_guess']
    job = cp2k(sett, plams.Molecule("test/test_files/C.xyz"))

    print("sett: ", sett)

    return run(job.energy)
