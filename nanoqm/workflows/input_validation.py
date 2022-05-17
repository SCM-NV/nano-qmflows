"""Functionality to check that the input provided by the user is valid.

Index
-----
.. currentmodule:: nanoqm.workflows.input_validation
.. autosummary:: process_input

API
---
.. autofunction:: process_input

"""

import os
import warnings
from os.path import join
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from schema import SchemaError
from scm.plams import Molecule

from qmflows import Settings
from qmflows.type_hints import PathLike

from .. import logger
from ..common import DictConfig, UniqueSafeLoader, valence_electrons
from .schemas import (schema_absorption_spectrum, schema_coop,
                      schema_cp2k_general_settings,
                      schema_derivative_couplings,
                      schema_distribute_absorption_spectrum,
                      schema_distribute_derivative_couplings,
                      schema_distribute_single_points, schema_ipr,
                      schema_single_points)
from .templates import create_settings_from_template

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


def process_input(input_file: PathLike, workflow_name: str) -> DictConfig:
    """Read the `input_file` in YAML format and validate it.

    Use the corresponding `workflow_name` schema and return a nested
    dictionary with the input.

    Parameters
    ----------
    input_file
        path to the input

    Returns
    -------
    dict
        Configuration to run the given workflow
    Raises
    ------
    SchemaError
        If the input is not valid

    """
    schema = schema_workflows[workflow_name]

    with open(input_file, 'r') as f:
        dict_input = yaml.load(f.read(), Loader=UniqueSafeLoader)

    try:
        d = schema.validate(dict_input)
        return DictConfig(InputSanitizer(d).sanitize())

    except SchemaError as e:
        msg = f"There was an error in the input yaml provided:\n{e}"
        logger.warning(msg)
        raise


class InputSanitizer:
    """Class to sanitize the input."""

    def __init__(self, input_dict: Dict):
        """Set the class properties."""
        self.user_input = input_dict
        self.general = input_dict["cp2k_general_settings"]

    def sanitize(self) -> Dict:
        """Apply all the sanity check on the input."""
        self.create_settings()
        self.apply_templates()
        self.add_missing_keywords()
        self.add_executable()
        self.print_final_input()

        return self.user_input

    def create_settings(self) -> None:
        """Transform the CP2K input dict into :class:`QMFLows.Settings`."""
        self.general['cp2k_settings_main'] = Settings(self.general['cp2k_settings_main'])
        self.general['cp2k_settings_guess'] = Settings(self.general['cp2k_settings_guess'])

    def apply_templates(self) -> None:
        """Apply a template for CP2K if the user requested it."""
        for s in [
                self.general[x] for x in ('cp2k_settings_main', 'cp2k_settings_guess')]:
            val = s['specific']

            if "template" in val:
                cp2k_template = create_settings_from_template(
                    self.general, val['template'], self.user_input["path_traj_xyz"])
                # remove template
                del s['specific']['template']

                # Add other keywords
                s['specific'] = cp2k_template.overlay(s['specific'])

    def add_executable(self) -> None:
        """Add executable to the job settings."""
        self.general['cp2k_settings_main']['executable'] = self.general['executable']
        self.general['cp2k_settings_guess']['executable'] = self.general['executable']

    def add_missing_keywords(self) -> None:
        """Add missing input data using the defaults."""
        # Add the `added_mos` and `mo_index_range` keywords
        if self.user_input.get('nHOMO') is None:
            self.user_input["nHOMO"] = self.compute_homo_index()

        # Added_mos keyword
        if self.user_input.get('compute_orbitals'):
            self.add_mo_index_range()
        else:
            logger.info("Orbitals are neither print nor store!")

        # Add restart point provided by the user
        self.add_restart_point()

        # Add basis sets
        self.add_basis()

        # add cell parameters
        self.add_cell_parameters()

        # Add Periodic properties
        self.add_periodic()

        # Add charge
        self.add_charge()

        # Add multiplicity
        self.add_multiplicity()

        # Add DFT exchange part
        self.add_functional_x()

        # Add DFT correlation part
        self.add_functional_c()

    def compute_homo_index(self) -> int:
        """Compute the index of the (doubly occupied) HOMO."""
        charge = self.general['charge']
        multiplicity = self.general['multiplicity']
        mol = Molecule(self.user_input["path_traj_xyz"], 'xyz')

        n_paired_electrons = sum(valence_electrons[at.symbol] for at in mol.atoms)
        n_paired_electrons -= charge  # Correct for total charge of the system
        n_paired_electrons -= (multiplicity - 1)  # Correct for the number of SOMOs

        assert (n_paired_electrons % 2) == 0
        return n_paired_electrons // 2

    def add_basis(self) -> None:
        """Add path to the basis and potential."""
        setts = [self.general[p] for p in ['cp2k_settings_main', 'cp2k_settings_guess']]

        root = os.path.abspath(self.general['path_basis'])
        if self.general['potential_file_name'] is not None:
            names = [join(root, f) for f in self.general["potential_file_name"]]
        else:
            names = [join(root, "GTH_POTENTIALS")]

        # add basis and potential path
        if self.general["path_basis"] is not None:
            logger.info("path to basis added to cp2k settings")
            for x in setts:
                x.basis = self.general['basis']
                x.potential = self.general['potential']

                # Do not overwrite explicitly specified CP2K settings
                dft = x.specific.cp2k.force_eval.dft
                if dft.get("potential_file_name") is None:
                    dft["potential_file_name"] = names

                # Choose the file basis to use
                self.select_basis_file(x)

    def select_basis_file(self, sett: Settings) -> None:
        """Choose the right basis set based on the potential and basis name."""
        dft = sett.specific.cp2k.force_eval.dft

        # Do not overwrite explicitly specified CP2K settings
        if dft.get("basis_set_file_name") is not None:
            return

        root = os.path.abspath(self.general['path_basis'])
        if self.general['basis_file_name'] is not None:
            dft["basis_set_file_name"] = [join(root, f) for f in self.general["basis_file_name"]]
        else:
            dft["basis_set_file_name"] = [join(root, "BASIS_MOLOPT")]
            # USE ADMM
            if dft.xc.get("xc_functional pbe") is None:
                dft["basis_set_file_name"] += [
                    join(root, "BASIS_ADMM_MOLOPT"),
                    join(root, "BASIS_ADMM"),
                ]

    def add_cell_parameters(self) -> None:
        """Add the Unit cell information to both the main and the guess settings."""
        # Search for a file containing the cell parameters
        for s in (self.general[p] for p in [
                'cp2k_settings_main',
                'cp2k_settings_guess']):
            if self.general["file_cell_parameters"] is None:
                s.cell_parameters = self.general['cell_parameters']
                s.cell_angles = None
            else:
                s.cell_parameters = None
            s.cell_angles = None

    def add_periodic(self) -> None:
        """Add the keyword for the periodicity of the system."""
        for s in (
            self.general[p] for p in [
                'cp2k_settings_main',
                'cp2k_settings_guess']):
            s.specific.cp2k.force_eval.subsys.cell.periodic = self.general['periodic']

    def add_charge(self) -> None:
        """Add the keyword for the charge of the system."""
        for s in (
            self.general[p] for p in [
                'cp2k_settings_main',
                'cp2k_settings_guess']):
            s.specific.cp2k.force_eval.dft.charge = self.general['charge']

    def add_multiplicity(self) -> None:
        """Add the keyword for the multiplicity of the system only if greater than 1."""
        if self.general['multiplicity'] > 1:
            for s in (
                    self.general[p] for p in ('cp2k_settings_main', 'cp2k_settings_guess')):
                s.specific.cp2k.force_eval.dft.multiplicity = self.general['multiplicity']
                s.specific.cp2k.force_eval.dft.uks = ""
        self.user_input["multiplicity"] = self.general["multiplicity"]

    def add_functional_x(self) -> None:
        """Add the keyword for the exchange part of the DFT functional: GGA or MGGA."""
        if self.general['functional_x'] is None:
            return
        for s in (
            self.general[p] for p in [
                'cp2k_settings_main',
                'cp2k_settings_guess']):
            s.specific.cp2k.force_eval.dft.xc.xc_functional[self.general['functional_x']] = {}

    def add_functional_c(self) -> None:
        """Add the keyword for the correlation part of the DFT functional: GGA or MGGA."""
        if self.general['functional_c'] is None:
            return
        for s in (
            self.general[p] for p in [
                'cp2k_settings_main',
                'cp2k_settings_guess']):
            s.specific.cp2k.force_eval.dft.xc.xc_functional[self.general['functional_c']] = {}

    def add_restart_point(self) -> None:
        """Add a restart file if the user provided it."""
        guess = self.general['cp2k_settings_guess']
        wfn = self.general['wfn_restart_file_name']
        if wfn is not None and wfn:
            dft = guess.specific.cp2k.force_eval.dft
            dft.wfn_restart_file_name = Path(wfn).absolute().as_posix()

    def add_mo_index_range(self) -> None:
        """Compute the MO range to print."""
        nocc, nvirt = self.user_input["active_space"]
        nSOMO = self.general["multiplicity"] - 1
        nHOMO = self.user_input["nHOMO"]

        mo_index_range = nHOMO - nocc, nHOMO + nSOMO + nvirt
        self.user_input["mo_index_range"] = mo_index_range

        # mo_index_range keyword
        cp2k_main = self.general['cp2k_settings_main']
        dft_main_print = cp2k_main.specific.cp2k.force_eval.dft.print
        dft_main_print.mo.mo_index_range = f"{mo_index_range[0] + 1} {mo_index_range[1]}"

        # added_mos
        dft = cp2k_main.specific.cp2k.force_eval.dft
        if nSOMO == 0:
            dft.scf.added_mos = mo_index_range[1] - nHOMO - nSOMO
        else:
            dft.scf.added_mos = f"{mo_index_range[1] - nHOMO - nSOMO} {mo_index_range[1] - nHOMO}"

        # Add section to Print the orbitals
        dft.print.mo.add_last = "numeric"
        dft.print.mo.each.qs_scf = 0
        dft.print.mo.eigenvalues = ""
        dft.print.mo.eigenvectors = ""
        dft.print.mo.ndigits = 36

    def print_final_input(self) -> None:
        """Print the input after post-processing."""
        xs = self.user_input.copy()

        for k, v in self.user_input.items():
            xs[k] = recursive_traverse(v)

        with open("input_parameters.yml", "w") as f:
            yaml.dump(xs, f, indent=4)


def recursive_traverse(val: Union[Dict, Settings, Any]) -> Union[Dict, Settings, Any]:
    """Check if the value of a key is a Settings instance a transform it to plain dict."""
    if isinstance(val, dict):
        if isinstance(val, Settings):
            return val.as_dict()
        else:
            return {k: recursive_traverse(v) for k, v in val.items()}
    else:
        return val
