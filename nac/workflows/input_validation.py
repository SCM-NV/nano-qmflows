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

        return create_settings(d)

    except SchemaError as e:
        msg = "There was an error in the input provided:\n{}".format(e)
        raise RuntimeError(msg)


def create_settings(d: Dict) -> Dict:
    """
    Transform the input dict into Cp2K settings and add the `added_mos` and `mo_index_range`
    keywords
    :param d: input dict
    :return: dictionary with Settings to call Cp2k
    """
    general = d['general_settings']

    # Convert cp2k definitions to settings
    general['settings_main'] = Settings(
        general['settings_main'])
    general['settings_guess'] = Settings(
        general['settings_guess'])

    # Add keywords if missing
    settings_main = general['settings_main']
    mo_index_range = general['mo_index_range']
    nHOMO = general["nHOMO"]
    dft = settings_main.specific.cp2k.force_eval.dft

    # Added_mos keyword
    scf = dft.get("scf", None)
    if scf is not None and scf.get("added_mos", None) is not None:
        dft.scf.added_mos = mo_index_range[1] - mo_index_range[0] - nHOMO + 1

    # mo_index_range keyword
    pr = dft.get("print", None)
    if pr is not None and pr.get('mo', None) is not None:
        pr.mo.mo_index_range = "{} {}".format(mo_index_range[0], mo_index_range[1])

    return d
