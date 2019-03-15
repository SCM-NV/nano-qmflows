#!/usr/bin/env python

from nac.workflows.input_validation import process_input
from nac.workflows import (workflow_stddft, workflow_derivative_couplings)
import argparse
import yaml

msg = "namd.py -i input"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-i', required=True,
                    help="Input file in YAML format")


dict_workflows = {'absorption_spectrum': workflow_stddft,
                  'derivative_couplings': workflow_derivative_couplings}


def main():
    input_file = read_cmd_line()
    with open(input_file, 'r') as f:
        dict_input = yaml.load(f, Loader=yaml.Loader)

    if 'workflow' not in dict_input:
        raise RuntimeError("The name of the workflow is required in the input file")
    else:
        workflow_name = dict_input['workflow']

    # Read and process input
    inp = process_input(input_file, workflow_name)

    # run workflow
    function = dict_workflows[workflow_name]
    function(inp)


def read_cmd_line():
    """
    Read the input file and the workflow name from the command line
    """
    args = parser.parse_args()
    return args.i


if __name__ == "__main__":
    main()
