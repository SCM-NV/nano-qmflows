#!/usr/bin/env python
"""Performs a molecular optimization using CP2K and prints local PDOS
projected on subsets of atoms based on the atom type and coordination number."""

from __future__ import annotations

import argparse
import itertools
import os
from typing import TYPE_CHECKING

from nanoCAT.recipes import coordination_number
from qmflows import Settings, cp2k, run, templates
from scm.plams import Molecule

from nanoqm import logger, __path__ as nanoqm_path
from nanoqm._logger import EnableFileHandler
from nanoqm.workflows.templates import generate_kinds

if TYPE_CHECKING:
    #: A nested dictonary
    NestedDict = dict[str, dict[int, list[int]]]

names = ("Ac", "MA")

molecules = {name: Molecule(f"{name}.xyz") for name in names}


def create_cp2k_settings(mol: Molecule) -> Settings:
    """Create CP2K general settings."""
    # Set path for basis set
    path_basis = os.path.join(nanoqm_path[0], "basis", "BASIS_MOLOPT")
    path_potential = os.path.join(nanoqm_path[0], "basis", "GTH_POTENTIALS")

    # Settings specifics
    s = Settings()
    s.basis = "DZVP-MOLOPT-SR-GTH"
    s.potential = "GTH-PBE"
    s.cell_parameters = 25
    s.specific.cp2k.force_eval.subsys.cell.periodic = 'none'
    s.specific.cp2k.force_eval.dft.basis_set_file_name = path_basis
    s.specific.cp2k.force_eval.dft.potential_file_name = path_potential

    # functional
    s.specific.cp2k.force_eval.dft.xc["xc_functional pbe"] = {}

    # Generate kinds for the atom types
    elements = [x.symbol for x in mol.atoms]
    kinds = generate_kinds(elements, s.basis, s.potential)

    # Update the setting with the kinds
    s.specific = s.specific + kinds

    return s


def compute_geo_opt(mol: Molecule, name: str, workdir: str, path_results: str) -> Molecule:
    """Perform the geometry optimization of **mol**."""
    # Get cp2k settings
    s = create_cp2k_settings(mol)

    # Update the setting with the geometry optimization template
    sett_geo_opt = templates.geometry.overlay(s)

    # Create the cp2k job
    opt_job = cp2k(sett_geo_opt, mol, job_name="cp2k_opt")

    # Run the cp2k job
    optimized_geometry = run(opt_job.geometry, path=workdir, folder=name)
    store_optimized_molecule(optimized_geometry, name, path_results)
    logger.info(f"{name} has been optimized with CP2K")

    return optimized_geometry


def store_optimized_molecule(optimized_geometry: Molecule, name: str, path_results: str) -> None:
    """Store the xyz molecular geometry."""
    path_geometry = f"{path_results}/{name}"
    if not os.path.exists(path_geometry):
        os.mkdir(path_geometry)
    with open(f"{path_geometry}/{name}_OPT.xyz", 'w', encoding="utf8") as f:
        optimized_geometry.writexyz(f)


def create_ldos_lists(coord: NestedDict) -> Settings:
    """Create CP2K settings for the LDOS section using the lists stored in **coord** dictionary."""
    # Settings specifics for PDOS section
    s = Settings()
    pdos = s.cp2k.force_eval.dft.print.pdos
    pdos["nlumo"] = 5

    # Possible repetitions of LDOS section
    chars = "ldos"
    combs = list(map(''.join, itertools.product(*zip(chars.upper(), chars.lower()))))

    # Filling the LDOS section with lists from dictionary
    coord_generator = (values for innerDict in coord.values() for values in innerDict.values())
    for i, v in enumerate(coord_generator):
        pdos[combs[i]]['list'] = ' '.join(map(str, v))

    return s


def compute_ldos(
    mol: Molecule,
    coord: NestedDict,
    name: str,
    workdir: str,
    path_results: str,
) -> None:
    """Compute the DOS projected on subsets of atoms given through lists.

    These lists are divided by atom type and coordination number.
    """
    # Get cp2k settings
    s = create_cp2k_settings(mol)

    # Get the cp2k ldos setting containing the lists
    ldos = create_ldos_lists(coord)

    # Join settings and update with the single point template
    s.specific = s.specific + ldos
    sett_ldos = templates.singlepoint.overlay(s)

    # Create the cp2k job
    dos_job = cp2k(sett_ldos, mol, job_name="cp2k_ldos")

    # Run the cp2k job
    run(dos_job, path=workdir, folder=name)
    store_coordination(coord, name, path_results)
    logger.info(f"{name} LDOS has been printed with CP2K")


def store_coordination(coord: NestedDict, name: str, path_results: str) -> None:
    """Store the LDOS lists informations."""
    tuple_generator = ((outerKey, innerKey, values) for outerKey, innerDict in coord.items()
                       for innerKey, values in innerDict.items())

    t = 'Atom  #Coord  List      Indices\n'
    for i, v in enumerate(tuple_generator, start=1):
        t += f'{v[0]}     {v[1]}      "list{i}"     {v[2]}\n'

    path_ldos = f"{path_results}/{name}"
    with open(f"{path_ldos}/coord_lists.out", 'w', encoding="utf8") as f:
        f.write(t)


@EnableFileHandler("output.log")
def main(workdir: str) -> None:
    # Create workdir if it doesn't exist (scratch)
    if not os.path.exists(workdir):
        os.mkdir(workdir)

    # Create folder to store the results
    path_results = "results"
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    # Compute the properties
    for name, mol in molecules.items():
        logger.info(f"computing: {name}")

        # Get the optimized geometries
        optimized_geometry = compute_geo_opt(mol, name, workdir, path_results)

        # Compute the coordination number of each atom
        coord = coordination_number(optimized_geometry)

        # Compute the DOS projected on lists
        compute_ldos(optimized_geometry, coord, name, workdir, path_results)


if __name__ == "__main__":
    """Parse the command line arguments and call the modeller class."""
    parser = argparse.ArgumentParser(
        description="opt_dos_multiple.py -w /home/test/workdir/")
    # configure logger
    parser.add_argument('-w', '--workdir', default="workflow")
    args = parser.parse_args()

    main(args.workdir)
