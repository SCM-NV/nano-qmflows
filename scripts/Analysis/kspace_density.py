
from cmath import (exp, pi, sqrt)
from functools import partial
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
    size = comm.Get_size()

    # Use only the root process to initialize all the variables
    atoms = readXYZ(path_xyz)
    symbols = np.array([at.symbol for at in atoms])
    if rank == 0:
        dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms)
    else:
        dictCGFs = None
    # Send the CGFs to all the process
    dictCGFs = comm.bcast(dictCGFs, root=0)

    # K-space grid to calculate the fuzzy band
    initial = (0., 0., 0.)  # Gamma point
    final = (0., 1., 1.)    # X point
    nPoints = 10
    grid_k_vectors = grid_kspace(initial, final, nPoints)

    # Calculate what part of the grid is computed by each process
    indexes = point_number_to_compute(size, nPoints)
    start, end = indexes[rank]

    # Each process extract a subset of the grid
    k_vectors = grid_k_vectors[start:end]

    # Fourier transform
    trans_mtx = read_hdf5_mpi(path_hdf5, join(project_name, 'trans_mtx'))

    # Compute the transformation in cartesian coordinates
    fun_fourier = partial(calculate_fourier_trasform_cartesian, symbols,
                          dictCGFs)

    # Apply the fourier transform then covert it to spherical and
    # sum all the transformations
    fun_reduce = lambda k: np.sum(np.dot(trans_mtx, fun_fourier(k)))

    # Apply the whole fourier transform to the subset of the grid
    # correspoding to each process
    partial_result = np.apply_along_axis(fun_reduce, 1, k_vectors)

    # Gather the results
    total_result = np.empty(nPoints)  # Array containing the final results

    # Index operation to gather the final vector
    indexes_trans = np.transpose(indexes)
    sendcounts = tuple(indexes_trans[0])
    displacements = tuple(indexes_trans[1] - indexes_trans[0])

    # Collective Gather using root
    comm.Gatherv(partial_result, total_result, sendcounts, displacements,
                 MPI.FLOAT, root=0)
    np.savetxt('grid.out', total_result)


def calculate_fourier_trasform_cartesian(atomic_symbols, dictCGFs, ks):
    """
    Calculate the Fourier transform projecting the MO in a set of plane waves

    mo_fourier(k) = < phi(r) | exp(i k . r)>

    :param atomic_symbols: Atomic symbols
    :type atomic_symbols: String
    :param ks: The vector in k-space where the fourier transform is evaluated.
    :type ks: Numpy Array
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)
    """
    dim_mo = np.sum(np.apply_along_axis(lambda s: len(dictCGFs[s]),
                                        0, atomic_symbols))
    molecular_orbital_transformed = np.empty(dim_mo)

    acc = 0
    for symb in atomic_symbols:
        num_CGFs = len(dictCGFs[symb])
        arr = calculate_fourier_trasform_atom(dictCGFs[symb], ks)
        molecular_orbital_transformed[acc: acc + num_CGFs] = arr
        acc += num_CGFs

    return molecular_orbital_transformed


def calculate_fourier_trasform_atom(cgfs, ks):
    """
    Calculate the Fourier transform for the set of CGFs in an Atom.
    """
    arr = np.empty(len(cgfs))
    for i, cgf in enumerate(cgfs):
        arr[i] = calculate_fourier_trasform_contracted(cgf, ks)

    return arr


def calculate_fourier_trasform_contracted(cgf, ks):
    """
    Compute the fourier transform for a given CGF.
    Implementation note: the function loops over the x,y and z coordinates
    while operate in the whole set of Contracted Gaussian primitives.
    """
    cs, es = cgf.primitives
    label = cgf.orbType
    mtx_cs_es = np.stack((cs, es))
    angular_momenta = compute_angular_momenta(label)
    acc = np.ones(3)
    for l, k in zip(angular_momenta, ks):
        fun_primitive = partial(calculate_fourier_trasform_primitive, l, k)
        rs = np.apply_along_axis(fun_primitive, 0, mtx_cs_es)
        acc *= rs

    return np.sum(acc)


def compute_angular_momenta(label):
    """
    Compute the exponents l,m and n for the CGF: x^l y^m z^n exp(-a (r-R)^2)
    """
    orbitalIndexes = {("S", 0): 0, ("S", 1): 0, ("S", 2): 0,
                      ("Px", 0): 1, ("Px", 1): 0, ("Px", 2): 0,
                      ("Py", 0): 0, ("Py", 1): 1, ("Py", 2): 0,
                      ("Pz", 0): 0, ("Pz", 1): 0, ("Pz", 2): 1,
                      ("Dxx", 0): 2, ("Dxx", 1): 0, ("Dxx", 2): 0,
                      ("Dxy", 0): 1, ("Dxy", 1): 1, ("Dxy", 2): 0,
                      ("Dxz", 0): 1, ("Dxz", 1): 0, ("Dxz", 2): 1,
                      ("Dyy", 0): 0, ("Dyy", 1): 2, ("Dyy", 2): 0,
                      ("Dyz", 0): 0, ("Dyz", 1): 1, ("Dyz", 2): 1,
                      ("Dzz", 0): 0, ("Dzz", 1): 0, ("Dzz", 2): 2}
    lookup = lambda i: orbitalIndexes[(label, i)]

    return np.apply_along_axis(lookup, 1, np.arange(3))


def calculate_fourier_trasform_primitive(l, k, c, alpha):
    """
    Compute the fourier transform for primitive Gaussian Type Orbitals.
    """
    pik2 = (pi ** 2) * (k ** 2)
    f = c * exp(- pik2 / alpha)
    if l == 0:
        return sqrt(pi / alpha) * f + 0j
    elif l == 1:
        im = k * (pi / alpha) ** f
        return complex(0, im)
    elif l == 2:
        return sqrt(pi / (alpha ** 5)) * (alpha / 2 - pik2) * f + 0j
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)


def read_hdf5_mpi(path_hdf5, comm, path_to_prop):
    """
    Read an array using the MPI interface of HDF5.
    """
    with h5py.File(path_hdf5, "r", driver="mpio", comm=comm) as f5:
        return f5[path_to_prop].value


def point_number_to_compute(size, points):
    """ Compute how many grid points is computed in a given mpi worker """
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


def grid_kspace(initial, final, points):
    """
    make a matrix of dimension Points x 3 containing the coordinates
    in the k-space to be sampled.
    """
    mtx = np.stack([np.linspace(i, f, points) for i, f in zip(initial, final)])

    return np.transpose(mtx)


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
    project_name = args.p
    path_hdf5 = args.hdf5
    path_xyz = args.xyz
    basis_name = args.basis

    return project_name, path_hdf5, path_xyz, basis_name


if __name__ == "__main__":
    main()
