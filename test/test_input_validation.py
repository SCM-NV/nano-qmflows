"""Test input validation functionality."""
import pytest
from qmflows import cp2k, run
from qmflows.type_hints import PathLike
from scm import plams

from nanoqm.common import read_cell_parameters_as_array
from nanoqm.workflows.input_validation import process_input

from .utilsTest import PATH_TEST, cp2k_available, remove_files


def test_input_validation() -> None:
    """Test the templates and keywords completion."""
    path_input = PATH_TEST / "input_test_pbe0.yml"
    dict_input = process_input(path_input, "derivative_couplings")
    sett = dict_input['cp2k_general_settings']['cp2k_settings_guess']

    scale_x = sett.specific.cp2k.force_eval.dft.xc.xc_functional.pbe.scale_x

    assert abs(scale_x - 0.75) < 1e-16


@pytest.mark.skipif(
    not cp2k_available(), reason="CP2K is not install or not loaded")
def test_call_cp2k_pbe() -> None:
    """Check if the input for a PBE cp2k job is valid."""
    try:
        results = run_plams(PATH_TEST / "input_test_pbe.yml")
        assert (results is not None)
    finally:
        remove_files()


def run_plams(path_input: PathLike) -> float:
    """Call Plams to run a CP2K job."""
    # create settings
    dict_input = process_input(path_input, "derivative_couplings")
    sett = dict_input['cp2k_general_settings']['cp2k_settings_guess']

    # adjust the cell parameters
    file_cell_parameters = dict_input['cp2k_general_settings'].get("file_cell_parameters")
    if file_cell_parameters is not None:
        array_cell_parameters = read_cell_parameters_as_array(file_cell_parameters)[1]
        sett.cell_parameters = array_cell_parameters[0, 2:11].reshape(3, 3).tolist()

        # Run the job
    job = cp2k(sett, plams.Molecule(PATH_TEST / "C.xyz"))

    return run(job.energy)
