
from cmath import (exp, pi, sqrt)
from mpi4py import MPI
from nac.schedule.components import create_dict_CGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ

import argparse
import h5py
import numpy as np
import os


def main():
    # Parse Command line
    project_name, path_hdf5, path_xyz, basis_name = read_cmd_line()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    atoms = readXYZ(path_xyz)
    
    if rank == 0:
        dictCGFs = create_dict_CGFs(path_hdf5, atoms, basis_name)

    trans_mtx = read_hdf5_mpi(path_hdf5, join(project_name, 'trans_mtx'))


def calculate_fourier_trasform_mo(k, atoms, dictCGFs):
    """
    Calculate the Fourier transform projecting the MO in a set of plane waves

    mo_fourier(k) = < phi(r) | exp(i k . r)>

    :param k: The vector in k-space where the fourier transform is evaluated.
    :type k: Numpy Array
    :param atoms: Molecular geometry
    :type atoms: [Namedtuple]
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)
    """


def calculate_fourier_trasform_primitive(l, c, alpha, k):
    pik2 = (pi ** 2) * (k ** 2)
    f = c * exp(- pik2 / alpha)
    if l == 0:
        return sqrt(pi / alpha) * f + 0j
    elif l == 1:
        im = k * (pi / alpha) ** f
        return complex(0, im)
    elif l == 2:
        return sqrt(pi / (alpha ** 5)) * (alpha / 2  - pik2) * f + 0j
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)


def read_hdf5_mpi(path_hdf5, comm, path_to_prop):
    """
    Read an array using the MPI interface of HDF5.
    """
    with  h5py.File(path_hdf5, "r", driver="mpio", comm=comm) as f5:
        return f5[path_to_prop].value


def createCGFs(path_hdf5, atoms, basis_name):
    """
    Create a dictionary containing the primitives Gaussian functions for
    each atom involved in the calculation.
    """
    home = os.path.expanduser('~')
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    cp2k_config = {"basis": basiscp2k, "potential": potcp2k}
    return create_dict_CGFs(path_hdf5, basis_name, atoms,
                            package_config=cp2k_config)

# Parser
msg = " script -hdf5 <path/to/hdf5> -xyz <path/to/geometry/xyz -b basis_name"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-p', required=True, help='Project name')
parser.add_argument('-hdf5', required=True, help='path to the HDF5 file')
parser.add_argument('-xyz', required=True, help='path to molecular gemetry')
parser.add_argument('-basis', required=True, help='Basis Name')


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    path_hdf5 = args.hdf5
    path_xyz = args.xyz
    basis_name = args.basis

    return path_hdf5, path_xyz, basis_name
