from .schemas import (
    schema_absorption_spectrum, schema_derivative_couplings,
    schema_electron_transfer, schema_general_settings)
from schema import SchemaError
from typing import Dict
import yaml
from qmflows.settings import Settings


schema_workflows = {
    'absorption_spectrum': schema_absorption_spectrum,
    'derivative_couplings': schema_derivative_couplings,
    'electron_transfer': schema_electron_transfer,
    'general_settings': schema_general_settings}


def process_input(input_file: str, workflow_name: str) -> Dict:
    """
    Read the `input_file` in YAML format, validate it against the
    corresponding `workflow_name` schema and return a nested dictionary with the input.

    :param str input_file: path to the input
    :return: Input as dictionary
    :raise SchemaError: If the input is not valid
    """
    schema = schema_workflows[workflow_name]

    with open(input_file, 'r') as f:
        dict_input = yaml.load(f.read())

    try:
        d = schema.validate(dict_input)

        # Convert cp2k definitions to settings
        d['general_settings']['settings_main'] = Settings(
            d['general_settings']['settings_main'])
        d['general_settings']['settings_guess'] = Settings(
            d['general_settings']['settings_guess'])

        return d

    except SchemaError as e:
        msg = "There was an error in the input provided:\n{}".format(e)
        raise RuntimeError(msg)
