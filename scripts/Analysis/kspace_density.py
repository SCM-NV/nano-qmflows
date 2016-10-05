
from functools import partial
from multiprocessing import Pool
from nac.integrals.fourierTransfrom import (calculate_fourier_trasform_cartesian,
                              fun_density_real, transform_to_spherical)
from nac.schedule.components import create_dict_CGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ

import argparse
import numpy as np
import os

# Some Hint about the types
Vector = np.ndarray
Matrix = np.ndarray


def main(parser):
    """
    These calculation is based on the paper:
    `Theoretical analysis of electronic band structure of
    2- to 3-nm Si nanocrystals`
    PHYSICAL REVIEW B 87, 195420 (2013)
    """
    # Parse Command line
    project_name, path_hdf5, path_xyz, basis_name, orbital = read_cmd_line(parser)
    # Use only the root process to initialize all the variables
    atoms = readXYZ(path_xyz)
    symbols = np.array([at.symbol for at in atoms])
    coords_angstrom = np.concatenate([at.xyz for at in atoms])
    au_to_angstrom = 1.889725989
    coords = au_to_angstrom * coords_angstrom

    # Dictionary containing as key the atomic symbols and as values the set of CGFs
    home = os.path.expanduser('~')
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    cp2k_config = {"basis": basiscp2k, "potential": potcp2k}
    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms,
                                package_config=cp2k_config)
    count_cgfs = np.vectorize(lambda s: len(dictCGFs[s]))
    number_of_basis = np.sum(np.apply_along_axis(count_cgfs, 0, symbols))

    # K-space grid to calculate the fuzzy band
    initial = (0., 1., 1.)  # Gamma point
    final = (0., 0., 0.)  # X point
    nPoints = 10
    grid_k_vectors = grid_kspace(initial, final, nPoints)

    # Calculate what part of the grid is computed by each process
    indexes = point_number_to_compute(1, nPoints)
    start, end = indexes[0]

    # Each process extract a subset of the grid
    k_vectors = grid_k_vectors[start:end]

    # Compute the fourier transformation in cartesian coordinates
    fun_fourier = partial(calculate_fourier_trasform_cartesian, symbols,
                          coords, dictCGFs, number_of_basis)

    # Apply the fourier transform then covert it to spherical
    fun_sphericals = partial(transform_to_spherical, fun_fourier,
                             path_hdf5, project_name, orbital)
    # Compute the momentum density (an Scalar)
    momentum_density = partial(fun_density_real, fun_sphericals)

    # Apply the whole fourier transform to the subset of the grid
    # correspoding to each process
    with Pool() as p:
        rss = p.map(momentum_density, k_vectors)

    # result = np.apply_along_axis(momentum_density, 1, k_vectors)
    print("Results: ", rss)


def point_number_to_compute(size, points) -> Vector:
    """ Compute how many grid points is computed in a given worker """
    res = points % size
    n = points // size

    def fun(rank):
        if res == 0:
            return n
        elif rank < res:
            return n + 1
        else:
            return n

    acc = 0
    xs = np.empty((size, 2), dtype=np.int)
    for i in range(size):
        dim = fun(i)
        xs[i] = acc, acc + dim
        acc += dim

    return xs


def grid_kspace(initial, final, points) -> Matrix:
    """
    make a matrix of dimension Points x 3 containing the coordinates
    in the k-space to be sampled.
    """
    mtx = np.stack([np.linspace(i, f, points) for i, f in zip(initial, final)])

    return np.transpose(mtx)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    project_name = args.p
    path_hdf5 = args.hdf5
    path_xyz = args.xyz
    basis_name = args.basis if args.basis is not None else "DZVP-MOLOPT-SR-GTH"
    orbital = args.orbital

    return project_name, path_hdf5, path_xyz, basis_name, orbital


if __name__ == "__main__":
    msg = " script -hdf5 <path/to/hdf5> -xyz <path/to/geometry/xyz -b basis_name"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True, help='Project name')
    parser.add_argument('-hdf5', required=True, help='path to the HDF5 file')
    parser.add_argument('-xyz', required=True, help='path to molecular gemetry')
    parser.add_argument('-basis', help='Basis Name')
    parser.add_argument('-orbital', required=True,
                        help='orbital to compute band', type=int)
    main(parser)
