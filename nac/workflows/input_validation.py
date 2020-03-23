"""Check that the input provided by the user is valid."""
import logging
import os
import warnings
from os.path import join
from pathlib import Path
from typing import Dict

import yaml
from schema import SchemaError
from scm.plams import Molecule

from nac.common import DictConfig
from qmflows.settings import Settings
from qmflows.type_hints import PathLike

from .schemas import (schema_absorption_spectrum, schema_coop,
                      schema_cp2k_general_settings,
                      schema_derivative_couplings,
                      schema_distribute_absorption_spectrum,
                      schema_distribute_derivative_couplings,
                      schema_distribute_single_points, schema_ipr,
                      schema_single_points)
from .templates import create_settings_from_template, valence_electrons

logger = logging.getLogger(__name__)


schema_workflows = {
    'absorption_spectrum': schema_absorption_spectrum,
    'derivative_couplings': schema_derivative_couplings,
    'single_points': schema_single_points,
    'cp2k_general_settings': schema_cp2k_general_settings,
    'distribute_derivative_couplings': schema_distribute_derivative_couplings,
    'distribute_absorption_spectrum': schema_distribute_absorption_spectrum,
    'distribute_single_points': schema_distribute_single_points,
    'ipr_calculation': schema_ipr,
    'coop_calculation': schema_coop
}


def process_input(input_file: PathLike, workflow_name: str) -> Dict:
    """Read the `input_file` in YAML format and validate it.

    Use the corresponding `workflow_name` schema and return a nested
    dictionary with the input.

    Parameters
    ----------
    input_file
        path to the input

    Raises
    ------
    SchemaError
        If the input is not valid
    """
    schema = schema_workflows[workflow_name]

    with open(input_file, 'r') as f:
        dict_input = yaml.load(f.read(), Loader=yaml.FullLoader)

    try:
        d = schema.validate(dict_input)
        return DictConfig(create_settings(d))

    except SchemaError as e:
        msg = f"There was an error in the input yaml provided:\n{e}"
        print(msg)
        raise


def create_settings(d: Dict) -> Dict:
    """Transform the input dict into Cp2K settings.

    :param d: input dict
    :return: dictionary with Settings to call Cp2k
    """
    # Convert cp2k definitions to settings
    general = d['cp2k_general_settings']
    general['cp2k_settings_main'] = Settings(
        general['cp2k_settings_main'])
    general['cp2k_settings_guess'] = Settings(
        general['cp2k_settings_guess'])

    apply_templates(general, d['path_traj_xyz'])

    input_parameters = add_missing_keywords(d)

    print_final_input(input_parameters)

    return input_parameters


def apply_templates(general: Dict, path_traj_xyz: str) -> None:
    """Apply a template for CP2K if the user request so."""
    for s in [
        general[x] for x in [
            'cp2k_settings_main',
            'cp2k_settings_guess']]:
        val = s['specific']

        if "template" in val:
            cp2k_template = create_settings_from_template(
                general, val['template'], path_traj_xyz)
            # remove template
            del s['specific']['template']
            # Add other keywords
            s['specific'] = cp2k_template.overlay(s['specific'])


def add_missing_keywords(d: Dict) -> Dict:
    """Add the `added_mos` and `mo_index_range` keywords."""
    general = d['cp2k_general_settings']
    # Add keywords if missing

    if d.get('nHOMO') is None:
        d['nHOMO'] = compute_HOMO_index(
            d['path_traj_xyz'],
            general['basis'],
            general['charge'])

    # Added_mos keyword
    if d.get('compute_orbitals'):
        add_mo_index_range(d)
    else:
        logger.info("Orbitals are neither print nor store!")

    # Add restart point
    add_restart_point(general)

    # Add basis sets
    add_basis(general)

    # add cell parameters
    add_cell_parameters(general)

    # Add Periodic properties
    add_periodic(general)

    # Add charge
    add_charge(general)

    # Add multiplicity
    add_multiplicity(general)

    return d


def add_basis(general: Dict) -> None:
    """Add path to the basis and potential."""
    setts = [general[p] for p in ['cp2k_settings_main', 'cp2k_settings_guess']]

    # add basis and potential path
    if general["path_basis"] is not None:
        logger.info("path to basis added to cp2k settings")
        for x in setts:
            x.basis = general['basis']
            x.potential = general['potential']
            x.specific.cp2k.force_eval.dft.potential_file_name = os.path.abspath(
                join(general['path_basis'], "GTH_POTENTIALS"))

            # Choose the file basis to use
            select_basis_file(x, general)


def select_basis_file(sett: Settings, general: Dict) -> str:
    """Choose the right basis set based on the potential and basis name."""
    dft = sett.specific.cp2k.force_eval.dft

    dft["basis_set_file_name"] = os.path.abspath(
        join(general['path_basis'], "BASIS_MOLOPT"))

    if dft.xc.get("xc_functional pbe") is None:
        # USE ADMM
        # We need to write one lowercase and the other uppercase otherwise Settings will
        # Overwrite the value
        dft["Basis_Set_File_Name"] = os.path.abspath(
            join(general['path_basis'], "BASIS_ADMM_MOLOPT"))
        dft["BASIS_SET_FILE_NAME"] = os.path.abspath(
            join(general['path_basis'], "BASIS_ADMM"))


def add_cell_parameters(general: Dict) -> None:
    """Add the Unit cell information to both the main and the guess settings."""
    # Search for a file containing the cell parameters
    file_cell_parameters = general["file_cell_parameters"]
    for s in (
        general[p] for p in [
            'cp2k_settings_main',
            'cp2k_settings_guess']):
        if file_cell_parameters is None:
            s.cell_parameters = general['cell_parameters']
            s.cell_angles = None
        else:
            s.cell_parameters = None
            s.cell_angles = None


def add_periodic(general: Dict) -> None:
    """Add the keyword for the periodicity of the system."""
    for s in (
        general[p] for p in [
            'cp2k_settings_main',
            'cp2k_settings_guess']):
        s.specific.cp2k.force_eval.subsys.cell.periodic = general['periodic']


def add_charge(general: Dict) -> None:
    """Add the keyword for the charge of the system."""
    for s in (
        general[p] for p in [
            'cp2k_settings_main',
            'cp2k_settings_guess']):
        s.specific.cp2k.force_eval.dft.charge = general['charge']


def add_multiplicity(general: Dict) -> None:
    """Add the keyword for the multiplicity of the system only if greater than 1."""
    if general['multiplicity'] > 1:
        for s in (
            general[p] for p in [
                'cp2k_settings_main',
                'cp2k_settings_guess']):
            s.specific.cp2k.force_eval.dft.multiplicity = general['multiplicity']
            s.specific.cp2k.force_eval.dft.uks = ""


def add_restart_point(general: Dict) -> None:
    """Add a restart file if the user provided it."""
    guess = general['cp2k_settings_guess']
    wfn = general['wfn_restart_file_name']
    if wfn is not None and wfn:
        dft = guess.specific.cp2k.force_eval.dft
        dft.wfn_restart_file_name = Path(wfn).absolute().as_posix()


def add_mo_index_range(dict_input: Dict) -> None:
    """Compute the MO range to print."""
    active_space = dict_input['active_space']
    nHOMO = dict_input["nHOMO"]
    mo_index_range = nHOMO - active_space[0], nHOMO + active_space[1]
    dict_input['mo_index_range'] = mo_index_range

    # mo_index_range keyword
    cp2k_main = dict_input['cp2k_general_settings']['cp2k_settings_main']
    dft_main_print = cp2k_main.specific.cp2k.force_eval.dft.print
    dft_main_print.mo.mo_index_range = "{} {}".format(
        mo_index_range[0] + 1, mo_index_range[1])

    # added_mos
    cp2k_main.specific.cp2k.force_eval.dft.scf.added_mos = mo_index_range[1] - nHOMO


def compute_HOMO_index(path_traj_xyz: str, basis: str, charge: int) -> int:
    """Compute the HOMO index."""
    mol = Molecule(path_traj_xyz, 'xyz')

    number_of_electrons = sum(
        valence_electrons['-'.join((at.symbol, basis))] for at in mol.atoms)

    # Correct for total charge of the system
    number_of_electrons = number_of_electrons - charge

    if (number_of_electrons % 2) != 0:
        warnings.warn(
            "Unpair number of electrons detected when computing the HOMO")

    return number_of_electrons // 2 + (number_of_electrons % 2)


def recursive_traverse(val):
    """Check if the value of a key is a Settings instance a transform it to plain dict."""
    if isinstance(val, dict):
        if isinstance(val, Settings):
            return val.as_dict()
        else:
            return {k: recursive_traverse(v) for k, v in val.items()}
    else:
        return val


def print_final_input(d: Dict) -> None:
    """Print the input after post-processing."""
    xs = d.copy()

    for k, v in d.items():
        xs[k] = recursive_traverse(v)

    with open("input_parameters.yml", "w") as f:
        yaml.dump(xs, f, indent=4)
