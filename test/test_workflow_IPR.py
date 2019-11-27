import os
import sys
from os.path import join
import pkg_resources as pkg
from nac.workflows.input_validation import process_input
from nac.workflows.workflow_ipr import workflow_ipr

# Environment data
file_path = pkg.resource_filename('nac', '')
root = os.path.split(file_path)[0]


def test_workflow_IPR(tmp_path):
    """Test the Inverse Participation Ratio workflow."""
    file_path = join(root, 'test/test_files/input_test_IPR.yml')
    config = process_input(file_path, 'ipr_calculation')
    print("config: ", config)
    try:
        result = workflow_ipr(config)
        print("IPR", result)
    except BaseException:
        print("scratch_path: ", tmp_path)
        print("Unexpected error:", sys.exc_info()[0])
        raise
