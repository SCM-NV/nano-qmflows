#!/usr/bin/env python
"""Perform a molecular optimization using a set of smiles and CP2K."""
import argparse
import os
import sys
from os.path import join

import pandas as pd
from noodles import gather
from scm.plams import Molecule, from_smiles

from nac.common import is_data_in_hdf5
from qmflows import Settings, cp2k, run, templates


def main(file_path: str):
    """Run script."""
    df = pd.read_csv(file_path)
    # HDF5 to store the arrays
    path_hdf5 = "database.h5"
    # Solvent to compute
    solvents = {"toluene": "cc1ccccc1", "o-xylene": "cc1c(c)cccc1",
     "octadecene": "C=CCCCCCCCCCCCCCCCC"}
    # Compute the properties
    # for smile in df["smiles"]:
    #     if is_data_in_hdf5(path_hdf5, )


if __name__ == "__main__":
    """Parse the command line arguments and call the modeller class."""
    parser = argparse.ArgumentParser(
        description="smiles_calculator.py -i smiles.csv")
    # configure logger
    parser.add_argument('-i', '--input', required=True,
                        help="Input file with the smiles")
    args = parser.parse_args()
    print(args)
    main(args.input)


# def create_cp2k_job(smile: str) -> object:
#     """Create a cp2k job for the given smile."""
#     systems = from_smiles(smile)

# # String representing the smile
# smiles = ['CCNCC', 'CCCNCCC', 'CCCCNCCCC', 'CCCCCNCCCCC', 'CCCCCCNCCCCCC',
#           'CCCCCCCNCCCCCCC', 'CCCCCCCCNCCCCCCCC', 'CCCCCCCCCNCCCCCCCCC',
#           'CCCCCCCCCCNCCCCCCCCCC', 'CCCCCCCCCCCNCCCCCCCCCCC',
#           'CCCCCCCCCCCCNCCCCCCCCCCCC', 'CCCCCCCCCCCCCNCCCCCCCCCCCCC',
#           'CCCCCCCCCCCCCCNCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCNCCCCCCCCCCCCCCC',
#           'CCCCCCCCCCCCCCCCNCCCCCCCCCCCCCCCC',
#           'CCCCCCCCCCCCCCCCCNCCCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCNCCCCCCCCCCCCCCCCC']

# systems = [from_smiles(s) for s in smiles]
# name_systems = ['C' + str(x) for x in range(2, 19)]

# # ======================
# # Settings for CP2k
# # ======================

# # Set path for basis set
# home = os.path.expanduser('~')
# basisCP2K = join(home, "cp2k_basis/BASIS_MOLOPT")
# potCP2K = join(home, "cp2k_basis/GTH_POTENTIALS")

# # Settings specifics
# s = Settings()
# s.basis = "DZVP-MOLOPT-SR-GTH"
# s.potential = "GTH-PBE"
# s.cell_parameters = 5
# s.specific.cp2k.force_eval.dft.basis_set_file_name = basisCP2K
# s.specific.cp2k.force_eval.dft.potential_file_name = potCP2K
# s.specific.cp2k.force_eval.subsys.cell.periodic = 'none'
# s.specific.cp2k.force_eval.subsys.kind["Cd"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q12"
# s.specific.cp2k.force_eval.subsys.kind["Cd"]["POTENTIAL"] = "GTH-PBE-q12"
# s.specific.cp2k.force_eval.subsys.kind["P"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q5"
# s.specific.cp2k.force_eval.subsys.kind["P"]["POTENTIAL"] = "GTH-PBE-q5"
# s.specific.cp2k.force_eval.subsys.kind["C"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q4"
# s.specific.cp2k.force_eval.subsys.kind["C"]["POTENTIAL"] = "GTH-PBE-q4"
# s.specific.cp2k.force_eval.subsys.kind["O"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q6"
# s.specific.cp2k.force_eval.subsys.kind["O"]["POTENTIAL"] = "GTH-PBE-q6"
# s.specific.cp2k.force_eval.subsys.kind["H"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q1"
# s.specific.cp2k.force_eval.subsys.kind["H"]["POTENTIAL"] = "GTH-PBE-q1"
# s.specific.cp2k.force_eval.subsys.kind["Se"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q6"
# s.specific.cp2k.force_eval.subsys.kind["Se"]["POTENTIAL"] = "GTH-PBE-q6"
# s.specific.cp2k.force_eval.subsys.kind["Cl"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q7"
# s.specific.cp2k.force_eval.subsys.kind["Cl"]["POTENTIAL"] = "GTH-PBE-q7"
# s.specific.cp2k.force_eval.dft.xc["xc_functional pbe"] = {}
# s.specific.cp2k.force_eval.subsys.kind["S"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q6"
# s.specific.cp2k.force_eval.subsys.kind["S"]["POTENTIAL"] = "GTH-PBE-q6"
# s.specific.cp2k.force_eval.subsys.kind["N"]["BASIS_SET"] = "DZVP-MOLOPT-SR-GTH-q5"
# s.specific.cp2k.force_eval.subsys.kind["N"]["POTENTIAL"] = "GTH-PBE-q5"

# # =======================
# # Optimize geometries with CP2k
# # =======================

# cp2k_jobs = [cp2k(templates.geometry.overlay(s), system, job_name=name_system)
#              for system, name_system in zip(systems, name_systems)]
# # =======================
# # Run the Calculations
# # =======================
# energies = gather(*[job.energy for job in cp2k_jobs])
# molecules = gather(*[job.geometry for job in cp2k_jobs])
# results_energies_cp2k, result_molecules_cp2k = run(
#     gather(energies, molecules), folder='/tmp/smiles')

# # ======================
# # Output the results
# # ======================

# for name, energy, mol in zip(systems, results_energies_cp2k, result_molecules_cp2k):
#     print("name: ", name)
#     print("energy: ", energy)
#     print("molecule: ", mol)


# == == == == == == == == == == == =
# Single Point Calculation with ADF on geometries optimized with CP2k
# == == == == == == == == == == == =
# settings_adf = Settings()
# settings_adf.input.basis.type = 'TZ2P'
# settings_adf.input.xc.gga = 'PBE'
# settings_adf.input.scf.converge = '1.0e-06'

# settings_crs = Settings()
# settings_crs.input.temperature = 298.15
# settings_crs.input.property._h = 'activitycoef'

# settings_adf.runscript.pre = "export SCM_TMPDIR=$PWD"
# settings_crs.runscript.pre = "export SCM_TMPDIR=$PWD"


# # ADF optimizations
# solvents = [Molecule('Toluene.xyz'), Molecule(
#     'Octadecene.xyz'), Molecule('DOE.xyz'), Molecule('o-xylene.xyz')]

# init('/home/frazac/ncligands/simplified/carboxilates')
# crs_dict = run_crs_adf(settings_adf, settings_crs,
#                        solvents, result_molecules_cp2k)
# finish()

# property_dict: dict = {name: [results.readkf('ACTIVITYCOEF', 'deltag')[1], results.readkf(
#     'ACTIVITYCOEF', 'gamma')[1]] for name, results in crs_dict.items()}

# df: pd.DataFrame = pd.DataFrame(property_dict).T
# df.columns = ['energy', 'activity coefficient']

# # Export the DataFrame to a .csv file
# filename: str = 'file.csv'
# df.to_csv(filename)
