#!/usr/bin/env python
"""Perform a molecular optimization using a set of smiles and CP2K."""
import argparse
import logging
import os
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import pkg_resources
from rdkit import Chem
from rdkit.Chem import AllChem
from scm.plams import (Molecule, finish, from_rdmol, from_smiles, init,
                       run_crs_adf)

from nac.workflows.templates import generate_kinds
from qmflows import Settings, cp2k, run, templates

# Starting logger
logger = logging.getLogger(__name__)

# Solvent to compute properties
solvents = OrderedDict({"toluene": "CC1=CC=CC=C1", "o-xylene": "CC1=CC=CC=C1C",
                        "octadecene": "CCCCCCCCCCCCCCCCC=C"})

Result = namedtuple("Result", ['gammas', 'delta_g'])


def set_logger():
    """Set logging default behaviour."""
    file_log = 'output.log'
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(asctime)s---%(levelname)s\n%(message)s\n',
                        datefmt='[%I:%M:%S]')


def create_multiindex_df() -> pd.DataFrame:
    """Create a multiindex DataFrame to store the compute properties for each smile/solvent."""
    idx = pd.MultiIndex.from_tuples([], names=["smile", "property"])

    return pd.DataFrame(columns=solvents.keys(), index=idx)


def store_results_in_df(smile: str, results: namedtuple, df_results: pd.DataFrame, path_results: str):
    """Store the computed properties in the results DataFrame."""
    cols = df_results.columns
    df_results.loc[(smile, "gammas"), cols] = results.gammas
    df_results.loc[(smile, "delta_g"), cols] = results.delta_g
    df_results.to_csv(path_results)


def compute_properties(smile: str) -> np.array:
    """Compute properties for the given smile and solvent."""
    # Create the CP2K job
    job_cp2k = create_job_cp2k(smile, smile)

    # Run the cp2k job
    optimized_geometry = run(job_cp2k.geometry, folder="/tmp/cp2k_job")

    # Create the ADF JOB
    crs_dict = create_job_adf(smile, optimized_geometry)

    # extract results
    delta_g = pd.Series({name: try_to_readkf("delta_g")
                         for name, results in crs_dict.items()})
    gammas = pd.Series({name: try_to_readkf("gamma")
                        for name, results in crs_dict.items()})

    return Result(gammas, delta_g)


def create_job_cp2k(smile: str, job_name: str) -> object:
    """Create a CP2K job object."""
    # Set path for basis set
    path_basis = pkg_resources.resource_filename("nac", "basis/BASIS_MOLOPT")
    path_potential = pkg_resources.resource_filename(
        "nac", "basis/GTH_POTENTIALS")

    # Settings specifics
    s = Settings()
    s.basis = "DZVP-MOLOPT-SR-GTH"
    s.potential = "GTH-PBE"
    s.cell_parameters = 5
    s.specific.cp2k.force_eval.dft.basis_set_file_name = path_basis
    s.specific.cp2k.force_eval.dft.potential_file_name = path_potential

    # functional
    s.specific.cp2k.force_eval.dft.xc["xc_functional"] = {}

    # Molecular geometry
    system = try_to_optimize(smile)

    # Generate kinds for the atom types
    elements = [x.symbol for x in system.atoms]
    kinds = generate_kinds(elements, s.basis, s.potential)

    # Update the setting with the kinds
    sett = templates.geometry.overlay(s)
    sett.specific = sett.specific + kinds

    return cp2k(sett, system, job_name="cp2k_opt")


def try_to_optimize(smile: str) -> Molecule:
    """Try to optimize the molecule with a force field."""
    try:
        # Try to optimize with RDKIT
        mol = Chem.MolFromSmiles(smile)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = from_rdmol(mol)
    except:
        mol = from_smiles(smile)

    return mol


def create_job_adf(smile: str, optimized_geometry: Molecule, workdir: str = "/tmp/adf") -> object:
    """Create a Single Point Calculation with ADF on geometries optimized with CP2k."""
    settings_adf = Settings()
    settings_adf.input.basis.type = 'TZ2P'
    settings_adf.input.xc.gga = 'PBE'
    settings_adf.input.scf.converge = '1.0e-06'

    settings_crs = Settings()
    settings_crs.input.temperature = 298.15
    settings_crs.input.property._h = 'activitycoef'

    settings_adf.runscript.pre = "export SCM_TMPDIR=$PWD"
    settings_crs.runscript.pre = "export SCM_TMPDIR=$PWD"

    # ADF optimizations
    solvents_geometries = [from_smiles(smile) for smile in solvents.values()]

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    init(workdir)
    crs_dict = run_crs_adf(settings_adf, settings_crs,
                           solvents_geometries, optimized_geometry)
    finish()

    return crs_dict


def try_to_readkf(results: object, property_name: str):
    """Try to read the output from a KF binary file."""
    try:
        return results.readkf("ACTIVITYCOEF", property_name)[1]
    except KeyError:
        return None


def main(file_path: str):
    """Run script."""
    set_logger()
    # Read input smiles
    df_smiles = pd.read_csv(file_path)
    # Path results
    path_results = "results.csv"
    # Read the database file o create new db
    df_results = pd.read_csv(path_results) if os.path.exists(
        path_results) else create_multiindex_df()

    # Compute the properties
    for smile in df_smiles["smiles"]:
        if smile in df_results.index:
            logger.info(f"properties of {smile} are already store!")
        else:
            results = compute_properties(smile)
            store_results_in_df(smile, results, df_results, path_results)


if __name__ == "__main__":
    """Parse the command line arguments and call the modeller class."""
    parser = argparse.ArgumentParser(
        description="smiles_calculator.py -i smiles.csv")
    # configure logger
    parser.add_argument('-i', '--input', required=True,
                        help="Input file with the smiles")
    args = parser.parse_args()

    main(args.input)
