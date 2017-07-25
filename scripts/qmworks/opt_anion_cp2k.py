#! /usr/bin/env python
"""
This programs performs a geometry optimization on the anion of a closed-shell
 molecule with cp2k using generic settings: DFT/PBE.

Note that is mandatory to define a cell_parameter, and a xyz structure.
If you have a restart file, a basis set and you can also define it in the command line.
It assumes that the basis and pot files are in $HOME/cp2k_basis
 folder in your home, which can be changed)
It assumes a DZVP by default, which can be also changed

It is always advised to submit the script using a JOB Manager like Slurm
"""

from qmworks import (cp2k, run, templates)
from scm.plams import Molecule
import argparse
from os.path import join
import os


def main(file_xyz, cell, restart, basis, basis_folder):
    """
    Define which systems need to be calculated
    """
    system = Molecule(file_xyz)

    # Set path for basis set
    basisCP2K = join(basis_folder, "BASIS_MOLOPT")
    potCP2K = join(basis_folder, "GTH_POTENTIALS")

    # Settings specifics
    s = templates.geometry
    s.basis = basis
    s.potential = "GTH-PBE"
    s.cell_parameters = cell
    s.specific.cp2k.force_eval.dft.basis_set_file_name = basisCP2K
    s.specific.cp2k.force_eval.dft.potential_file_name = potCP2K
    s.specific.cp2k.force_eval.dft.uks = ''
    s.specific.cp2k.force_eval.dft.charge = '-1'
    s.specific.cp2k.force_eval.dft.multiplicity = '2'
    s.specific.cp2k.force_eval.dft.wfn_restart_file_name = '{}'.format(restart)

# =======================
# Compute OPT files with CP2k
# =======================

    result = run(cp2k(s, system))

# ======================
# Output the results
# ======================

    print(result.energy)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()

    attributes = ['xyz', 'cell', 'restart', 'basis', 'bas_fold']

    return [getattr(args, p) for p in attributes]

# ============<>===============
if __name__ == "__main__":

    msg = "plot_decho -xyz <path/to/xyz> -cell <cell_size>\
    -restart <path/to/restart_file_name>\
    -basis <nameOfBasisSet>\
    -bas_fold <path/to/basis_set>"

    home = os.path.expanduser('~')

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        '-xyz', required=True, help='path to the xyz file')
    parser.add_argument(
        '-cell', required=True, help='Size of the cell')
    parser.add_argument(
        '-restart', type=str, default='', help='path to restart file name')
    parser.add_argument(
        '-basis', type=str, default='DZVP-MOLOPT-SR-GTH', help='Basis-set name')
    parser.add_argument(
        '-bas_fold', type=str, default=join(home, 'cp2k_basis'),
        help='Location of basis set files')

    main(*read_cmd_line(parser))
