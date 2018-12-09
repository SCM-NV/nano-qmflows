from nac.workflows.input_validation import process_input
from qmflows import (cp2k, run)
from scm import plams
import pytest


@pytest.mark.skip
def test_input_validation():
    """
    test the templates and keywords completion
    """
    path_input = "test/test_files/input_test_templates.yml"
    dict_input = process_input(path_input, "absorption_spectrum")
    sett = dict_input['general_settings']['settings_main']

    job = cp2k(sett, plams.Molecule("test/test_files/Cd.xyz"))

    results = run(job.energy)

    print(results)
