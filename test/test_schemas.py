"""Check the schemas."""
from assertionlib import assertion

from nanoqm.workflows.input_validation import process_input

from .utilsTest import PATH_TEST


def test_input_validation():
    """Test the input validation schema."""
    schemas = ("absorption_spectrum", "derivative_couplings")
    paths = [PATH_TEST / x for x in
             ["input_test_absorption_spectrum.yml", "input_fast_test_derivative_couplings.yml"]]
    for s, p in zip(schemas, paths):
        d = process_input(p, s)
        assertion.isinstance(d, dict)
