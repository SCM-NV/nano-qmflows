#!/usr/bin/env python
"""Comman line interface to run the workflows.

Usage:
    run_workflow.py -i input.yml

Available workflow:
    * absorption_spectrum
    * derivative_couplings
    * single_points
    * ipr_calculation
    * coop_calculation

"""

import argparse
import logging
import os

import yaml

from ..common import UniqueSafeLoader
from .input_validation import process_input
from .workflow_coop import workflow_crystal_orbital_overlap_population
from .workflow_coupling import workflow_derivative_couplings
from .workflow_ipr import workflow_ipr
from .workflow_single_points import workflow_single_points
from .workflow_stddft_spectrum import workflow_stddft

logger = logging.getLogger(__name__)

msg = "run_workflow.py -i input.yml"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-i', required=True,
                    help="Input file in YAML format")


dict_workflows = {
    'absorption_spectrum': workflow_stddft,
    'derivative_couplings': workflow_derivative_couplings,
    'single_points': workflow_single_points,
    'ipr_calculation': workflow_ipr,
    'coop_calculation': workflow_crystal_orbital_overlap_population}


def main():
    """Parse the command line arguments and run workflow."""
    args = parser.parse_args()
    input_file = args.i
    with open(input_file, 'r') as f:
        dict_input = yaml.load(f, Loader=UniqueSafeLoader)
    if 'workflow' not in dict_input:
        raise RuntimeError(
            "The name of the workflow is required in the input file")
    else:
        workflow_name = dict_input['workflow']

    # Read and process input
    inp = process_input(input_file, workflow_name)

    # run workflow
    function = dict_workflows[workflow_name]

    logger.info(f"Running worflow using: {os.path.abspath(input_file)}")
    function(inp)


if __name__ == "__main__":
    main()
