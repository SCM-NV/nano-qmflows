"""Test input validation functionality."""

import os

import yaml
import pytest
from qmflows import cp2k, run
from qmflows.type_hints import PathLike
from qmflows.packages.cp2k_package import CP2K_Result
from scm import plams

import nanoqm
from nanoqm.common import read_cell_parameters_as_array
from nanoqm.workflows.input_validation import process_input, schema_workflows, InputSanitizer

from .utilsTest import PATH_TEST, cp2k_available, remove_files, validate_status


class TestInputValidation:
    def test_basic(self) -> None:
        """Test the templates and keywords completion."""
        path_input = PATH_TEST / "input_test_pbe0.yml"
        dict_input = process_input(path_input, "derivative_couplings")
        sett = dict_input['cp2k_general_settings']['cp2k_settings_guess']

        scale_x = sett.specific.cp2k.force_eval.dft.xc.xc_functional.pbe.scale_x

        assert abs(scale_x - 0.75) < 1e-16

    @pytest.mark.parametrize("key", ["potential_file_name", "basis_set_file_name"])
    def test_filename_override(self, key: str) -> None:
        """Test that filename overrides are respected."""
        with open(PATH_TEST / "input_test_pbe0.yml", "r") as f:
            s = plams.Settings(yaml.load(f, Loader=yaml.SafeLoader))
            dft1 = s.cp2k_general_settings.cp2k_settings_main.specific.cp2k.force_eval.dft
            dft1[key] = "test"

        s_output = schema_workflows["derivative_couplings"].validate(s)
        InputSanitizer(s_output).sanitize()

        dft2 = s_output.cp2k_general_settings.cp2k_settings_main.specific.cp2k.force_eval.dft
        assert dft2[key] == "test"

    @pytest.mark.parametrize("is_list", [True, False], ids=["list", "str"])
    @pytest.mark.parametrize("key", ["potential_file_name", "basis_file_name"])
    def test_basis_filename(self, key: str, is_list: bool) -> None:
        with open(PATH_TEST / "input_test_pbe0.yml", "r") as f:
            s = plams.Settings(yaml.load(f, Loader=yaml.SafeLoader))
            dft1 = s.cp2k_general_settings[key] = ["a", "b", "c"] if is_list else "a"

        s_output = schema_workflows["derivative_couplings"].validate(s)
        InputSanitizer(s_output).sanitize()

        root = os.path.join(nanoqm.__path__[0], "basis")
        if is_list:
            ref = [os.path.join(root, i) for i in ["a", "b", "c"]]
        else:
            ref = [os.path.join(root, "a")]
        dft2 = s_output.cp2k_general_settings.cp2k_settings_main.specific.cp2k.force_eval.dft
        assert dft2["basis_set_file_name" if key == "basis_file_name" else key] == ref

    def test_basis(self) -> None:
        with open(PATH_TEST / "input_test_pbe0.yml", "r") as f:
            s = plams.Settings(yaml.load(f, Loader=yaml.SafeLoader))
            s.cp2k_general_settings.basis = "DZVP-MOLOPT-MGGA-GTH"
            s.cp2k_general_settings.potential = "GTH-MGGA"

        s_out = schema_workflows["derivative_couplings"].validate(s)
        InputSanitizer(s_out).sanitize()

        ref = {
            "C": {
                "basis_set": ["DZVP-MOLOPT-MGGA-GTH-q4", "AUX_FIT CFIT3"],
                "potential": "GTH-MGGA-q4",
            },
            "H": {
                "basis_set": ["DZVP-MOLOPT-MGGA-GTH-q1", "AUX_FIT CFIT3"],
                "potential": "GTH-MGGA-q1",
            },
        }
        kind = s_out.cp2k_general_settings.cp2k_settings_main.specific.cp2k.force_eval.subsys.kind
        assert kind == ref

    @pytest.mark.parametrize("functional_c", [True, False])
    @pytest.mark.parametrize("functional_x", [True, False])
    def test_functional(self, functional_c: bool, functional_x: bool) -> None:
        with open(PATH_TEST / "input_test_pbe0.yml", "r") as f:
            s = plams.Settings(yaml.load(f, Loader=yaml.SafeLoader))
            s.cp2k_general_settings.cp2k_settings_main.specific.template = "main"
            if functional_c:
                s.cp2k_general_settings["functional_c"] = "GGA_C_LYP"
            if functional_x:
                s.cp2k_general_settings["functional_x"] = "GGA_X_OL2"

        s_out = schema_workflows["derivative_couplings"].validate(s)
        InputSanitizer(s_out).sanitize()

        ref = {}
        if functional_c:
            ref["GGA_C_LYP"] = {}
        if functional_x:
            ref["GGA_X_OL2"] = {}

        dft = s_out.cp2k_general_settings.cp2k_settings_main.specific.cp2k.force_eval.dft
        assert dft.xc.xc_functional == ref


@pytest.mark.skipif(
    not cp2k_available(), reason="CP2K is not install or not loaded")
def test_call_cp2k_pbe() -> None:
    """Check if the input for a PBE cp2k job is valid."""
    try:
        result = run_plams(PATH_TEST / "input_test_pbe.yml")
        validate_status(result)
        assert result.energy is not None
    finally:
        remove_files()


def run_plams(path_input: PathLike) -> CP2K_Result:
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

    return run(job)
