
from cmath import (exp, pi, sqrt)
from functools import partial
from nac.basisSet.basisNormalization import createNormalizedCGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ


import argparse
import h5py
import numpy as np
import os


# Some Hint about the types
from typing import Callable, Dict, NamedTuple
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
    
    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms)
    # K-space grid to calculate the fuzzy band
    initial = (0., 0., 0.)  # Gamma point
    final = (0., 1., 1.)    # X point
    nPoints = 20
    grid_k_vectors = grid_kspace(initial, final, nPoints)

    # Calculate what part of the grid is computed by each process
    indexes = point_number_to_compute(1, nPoints)
    start, end = indexes[0]

    # Each process extract a subset of the grid
    k_vectors = grid_k_vectors[start:end]

    # Compute the fourier transformation in cartesian coordinates
    fun_fourier = partial(calculate_fourier_trasform_cartesian, symbols,
                          coords, dictCGFs)

    # Apply the fourier transform then covert it to spherical
    fun_sphericals = lambda k: transform_to_spherical(fun_fourier,
                                                      path_hdf5,
                                                      project_name,
                                                      orbital, k)
    # Compute the momentum density (an Scalar)
    momentum_density = lambda k: fun_density_real(fun_sphericals, k)

    # Apply the whole fourier transform to the subset of the grid
    # correspoding to each process
    result = np.apply_along_axis(momentum_density, 1, k_vectors)
    print("Results: ", result)

def fun_density_real(function: Callable, k: float) -> float:
    """ Compute the momentum density"""
    xs = function(k)
    print("Orbital transformation is: ", xs)
    return np.dot(xs, np.conjugate(xs)).real


def transform_to_spherical(fun_fourier: Callable, path_hdf5: str,
                           project_name: str, orbital: str,
                           k: Vector) -> complex:
    """
    Calculate the Fourier transform in Cartesian, convert it to Spherical
    multiplying by the `trans_mtx` and finally multiply the coefficients
    in Spherical coordinates.
    """
    trans_mtx = read_hdf5(path_hdf5, join(project_name, 'trans_mtx'))
    path_to_mo = join(project_name, 'point_0/cp2k/mo/coefficients')
    molecular_orbital_i = read_hdf5(path_hdf5, path_to_mo)[:, orbital]

    return np.dot(molecular_orbital_i, np.dot(trans_mtx, fun_fourier(k)))


def calculate_fourier_trasform_cartesian(atomic_symbols: Vector,
                                         atomic_coords: Vector,
                                         dictCGFs: Dict,
                                         ks: Vector) -> Vector:
    """
    Calculate the Fourier transform projecting the MO in a set of plane waves

    mo_fourier(k) = < phi(r) | exp(i k . r)>

    :param atomic_symbols: Atomic symbols
    :type atomic_symbols: Numpy Array [String]
    :param ks: The vector in k-space where the fourier transform is evaluated.
    :type ks: Numpy Array
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)

    returns: Numpy array
    """
    print("K-vector: ", ks)
    fun = np.vectorize(lambda s: len(dictCGFs[s]))
    dim_mo = np.sum(np.apply_along_axis(fun,  0, atomic_symbols))
    molecular_orbital_transformed = np.empty(int(dim_mo), dtype=np.complex128)

    acc = 0
    for i, symb in enumerate(atomic_symbols):
        num_CGFs = len(dictCGFs[symb])
        i3, i3_1 = i * 3, 3 * (i + 1)
        xyz = atomic_coords[i3: i3_1]
        arr = calculate_fourier_trasform_atom(dictCGFs[symb], xyz, ks)
        molecular_orbital_transformed[acc: acc + num_CGFs] = arr
        acc += num_CGFs

    return molecular_orbital_transformed


def calculate_fourier_trasform_atom(cgfs: Dict, xyz: Vector,
                                    ks: Vector) -> Vector:
    """
    Calculate the Fourier transform for the set of CGFs in an Atom.
    """
    arr = np.empty(len(cgfs), dtype=np.complex128)
    for i, cgf in enumerate(cgfs):
        arr[i] = calculate_fourier_trasform_contracted(cgf, xyz, ks)

    return arr


def calculate_fourier_trasform_contracted(cgf: NamedTuple, xyz: Vector,
                                          ks: Vector) -> complex:
    """
    Compute the fourier transform for a given CGF.
    Implementation note: the function loops over the x,y and z coordinates
    while operate in the whole set of Contracted Gaussian primitives.
    """
    cs, es = cgf.primitives
    label = cgf.orbType
    angular_momenta = compute_angular_momenta(label)
    acc = np.ones(cs.shape, dtype=np.complex128)

    # Accumlate x, y and z for each one of the primitves
    for l, x, k in zip(angular_momenta, xyz, ks):
        fun_primitive = partial(calculate_fourier_trasform_primitive, l, x, k)
        rs = np.apply_along_axis(np.vectorize(fun_primitive), 0, es)
        acc *= rs 

    # The result is the summation of the primitive multiplied by is corresponding
    # coefficients
    return np.dot(acc, cs)


def compute_angular_momenta(label) -> Vector:
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

    return np.apply_along_axis(np.vectorize(lookup), 0, np.arange(3))


def calculate_fourier_trasform_primitive(l: int, x: float, k: float,
                                         alpha: float) -> complex:
    """
    Compute the fourier transform for primitive Gaussian Type Orbitals.
    """
    pik = pi * k
    f = exp(-alpha * x ** 2 + complex(alpha * x, - pik) ** 2 / alpha)
    if l == 0:
        return sqrt(pi / alpha) * f
    elif l == 1:
        f = k * exp(-pik * complex(pik / alpha, 2 * x))
        return (pi / alpha) ** 1.5  * f
    elif l == 2:
        f = exp(-pik * complex(pik / alpha, 2 * x))
        return sqrt(pi / (alpha ** 5)) * (alpha / 2 - pik ** 2) * f
    else:
        msg = ("there is not implementation for the primivite fourier "
               "transform of l: {}".format(l))
        raise NotImplementedError(msg)


def read_hdf5(path_hdf5, path_to_prop):
    """
    Read an array using the MPI interface of HDF5.
    """
    with h5py.File(path_hdf5, "r") as f5:
        return f5[path_to_prop].value


def point_number_to_compute(size, points) -> Vector:
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


def grid_kspace(initial, final, points) -> Matrix:
    """
    make a matrix of dimension Points x 3 containing the coordinates
    in the k-space to be sampled.
    """
    mtx = np.stack([np.linspace(i, f, points) for i, f in zip(initial, final)])

    return np.transpose(mtx)


def createCGFs(path_hdf5, atoms, basis_name) -> Dict:
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


def create_dict_CGFs(path_hdf5, basisname, xyz, package_name='cp2k',
                     package_config=None):
    """
    Try to read the basis from the HDF5 otherwise read it from a file and store
    it in the HDF5 file. Finally, it reads the basis Set from HDF5 and calculate
    the CGF for each atom.

    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    type path_hdf5: String
    :param basisname: Name of the Gaussian basis set.
    :type basisname: String
    :param xyz: List of Atoms.
    :type xyz: [nac.common.AtomXYZ]
    """
    with h5py.File(path_hdf5, "r") as f5:
        return createNormalizedCGFs(f5, basisname, package_name, xyz)


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
