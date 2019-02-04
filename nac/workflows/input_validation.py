from .schemas import (
    schema_absorption_spectrum, schema_distribute_derivative_couplings,
    schema_derivative_couplings, schema_electron_transfer, schema_cp2k_general_settings)
from .templates import (
    cp2k_pbe0_guess, cp2k_pbe0_main, cp2k_pbe_guess, cp2k_pbe_main)
from qmflows.settings import Settings
from schema import SchemaError
from typing import Dict
import logging
import os
import yaml

logger = logging.getLogger(__name__)

schema_workflows = {
    'absorption_spectrum': schema_absorption_spectrum,
    'derivative_couplings': schema_derivative_couplings,
    'electron_transfer': schema_electron_transfer,
    'cp2k_general_settings': schema_cp2k_general_settings,
    'distribute_derivative_couplings': schema_distribute_derivative_couplings}


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
    Transform the input dict into Cp2K settings.

    :param d: input dict
    :return: dictionary with Settings to call Cp2k
    """
    # Convert cp2k definitions to settings
    general = d['cp2k_general_settings']
    general['cp2k_settings_main'] = Settings(
        general['cp2k_settings_main'])
    general['cp2k_settings_guess'] = Settings(
        general['cp2k_settings_guess'])

    d = apply_templates(d)

    return add_missing_keywords(d)


def apply_templates(d: Dict):
    """
    Apply a template for CP2K if the user request so.
    """
    general = d['cp2k_general_settings']

    # available templates
    templates_dict = {
        "pbe_guess": cp2k_pbe_guess, "pbe_main": cp2k_pbe_main,
        "pbe0_guess": cp2k_pbe0_guess, "pbe0_main": cp2k_pbe0_main}

    for s in [general[x] for x in ['cp2k_settings_main', 'cp2k_settings_guess']]:
        val = s['specific']

        if "template" in val:
            s['specific'] = templates_dict[val['template']]
    return d


def add_missing_keywords(d: Dict) -> Dict:
    """
    and add the `added_mos` and `mo_index_range` keywords
    """
    general = d['cp2k_general_settings']
    # Add keywords if missing

    # Added_mos keyword
    add_mo_index_range(general)

    # Add restart point
    add_restart_point(general)

    # Add basis sets
    add_basis(general)

    # add cell parameters
    add_cell_parameters(general)

    return d


def add_basis(general: dict) -> None:
    """
    Add path to the basis and potential
    """
    setts = [general[p] for p in ['cp2k_settings_main', 'cp2k_settings_guess']]

    # add basis and potential path
    if all(general[x] is not None for x in ["path_basis", "path_potential"]):
        logger.info("path_basis and path_potential added to cp2k settings")
        for x in setts:
            x.basis = general['basis']
            x.potential = general['potential']
            x.specific.cp2k.force_eval.dft.basis_set_file_name = os.path.abspath(
                general['path_basis'])
            x.specific.cp2k.force_eval.dft.potential_file_name = os.path.abspath(
                general['path_potential'])


def add_cell_parameters(general: dict) -> None:
    """
    Add the Unit cell information to both the main and the guess settings
    """
    for s in (general[p] for p in ['cp2k_settings_main', 'cp2k_settings_guess']):
        s.cell_parameters = general['cell_parameters']
        s.cell_angles = general['cell_angles']


def add_restart_point(general: dict) -> None:
    """
    """
    guess = general['cp2k_settings_guess']
    wfn = guess['wfn_restart_file_name']
    if wfn is not None and wfn:
        dft = guess.specific.cp2k.force_eval.dft
        dft.wfn_restart_file_name = wfn


def add_mo_index_range(general: dict) -> None:
    """
    Compute the MO range to print
    """
    mo_index_range = compute_mo_index_range()

    # mo_index_range keyword
    dft_main_print = general['cp2k_settings_main'].specific.cp2k.force_eval.dft.print
    dft_main_print.mo.mo_index_range = "{} {}".format(mo_index_range[0], mo_index_range[1])


def compute_mo_index_range():
    """ """
    pass
