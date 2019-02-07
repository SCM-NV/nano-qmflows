
from nac.workflows.input_validation import process_input
from os.path import join


def test_input_validation():
    """
    Test the input validation schema
    """
    root = "test/test_files"
    schemas = ("absorption_spectrum", "derivative_couplings")
    paths = [join(root, x) for x in
             ["input_test_absorption_spectrum.yml", "input_test_derivative_couplings.yml"]]
    for s, p in zip(schemas, paths):
        d = process_input(p, s)
        assert isinstance(d, dict)
