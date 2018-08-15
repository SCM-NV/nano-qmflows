
from distutils.spawn import find_executable
from functools import partial
from multiprocessing import (cpu_count, Pool)
from nac.common import (Matrix, Vector)

from multipoleObaraSaika import sab_multipole  # compiled with cython
from subprocess import (PIPE, Popen)
from typing import (Callable, Dict, List, Tuple)
import dill
import numpy as np
import os
import tempfile
import warnings

try:
    from dask.distributed import Client
except ImportError:
    msg = """The dask and distributed libraries must be installed if you want to use dask
    to distribute the computation"""
    warnings.warn(msg)


def general_multipole_matrix(
        molecule: List, dictCGFs: List,
        calculator: Callable=None, runner='multiprocessing',
        ncores: int=None) -> Vector:
    """
    Generic function to calculate a matrix using a Gaussian basis set and
    the molecular geometry.
    Build a matrix using a pool of worker and a function takes nuclear
    corrdinates and a Contracted Gauss function and compute a number.

    :param molecule: Atomic label and cartesian coordinates.
    :param dictCGFs: Contracted gauss functions normalized, represented as
    a dict of list containing the Contracted Gauss primitives
    :param calculator: Function to compute the matrix elements.
    :param runner: distributed system to compute the elements of the matrix
    :param ncores: number of available cores
    :returns: Numpy Array representing a flatten triangular matrix.
    """
    # Indices of the cartesian coordinates and corresponding CGFs
    indices, nOrbs = compute_CGFs_indices(molecule, dictCGFs)
    function = partial(calculator, molecule, dictCGFs, indices)
    ncores = ncores if ncores is not None else cpu_count()

    if runner.lower() == 'dask':
        return runner_dask(function, nOrbs)
    elif runner.lower() == 'mpi':
        return runner_mpi(function, nOrbs, ncores)
    else:
        # Create a list of indices of a triangular matrix to distribute
        # the computation of the matrix uniformly among the available cores
        block_triang_indices = compute_block_triang_indices(nOrbs, ncores)
        rss = runner_multiprocessing(function, block_triang_indices)

        return np.concatenate(rss)


def runner_dask(
        function: Callable, nOrbs: int, ncores: int=None) -> Matrix:
    """
    Use the Dask library to distribute the computation of the integrals.

    :param function: callable to compute the multipole matrix.
    """
    # setup cluster
    client = Client()
    client.cluster
    ncores = len(client.ncores())

    block_triang_indices = compute_block_triang_indices(nOrbs, ncores)
    matrix = client.map(function, block_triang_indices)

    return np.concatenate(client.gather(matrix))


def runner_multiprocessing(
        function: Callable, indices_chunks: List) -> Matrix:
    """
    Compute a multipole matrix using the python multiprocessing module.

    :param function: callable to compute the multipole matrix.
    :param indices_chunks: List of Matrices/tuples containing the indices
    compute for each core/worker.
    :returns: multipole matrix.
    """
    with Pool() as p:
        rss = p.map(function, indices_chunks)

    return rss


def runner_mpi(
        function: Callable, nOrbs: int, ncores: int=None) -> Matrix:
    """
    Compute a multipole matrix using the python using mpi.
    It spawn a subprocess that calls MPI and store the resulting matrix in a numpy
    file.

    :param function: callable to compute the multipole matrix.
    :param nOrbs: number of CGFs for the whole molecule.
    :returns: multipole matrix.
    """
    # Serialize the partial applied function
    tmp_fun = tempfile.mktemp(prefix='serialized_multipole', suffix='.dill', dir='.')
    with open(tmp_fun, 'wb') as f:
        dill.dump(function, f)

    # Run the MPI command
    tmp_out = tempfile.mktemp(prefix='mpi_multipole_', dir='.')
    executable = find_executable('call_mpi_multipole.py')
    cmd = "mpiexec -n {} python {} -f {} -n {} -o {}".format(
        ncores, executable, tmp_fun, nOrbs, tmp_out)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    rs = p.communicate()
    err = rs[1]
    if err:
        raise RuntimeError("Submission Errors: {}".format(err))
        clean([output_filename, tmp_fun])
    else:
        output_filename = tmp_out + '.npy'
        arr = np.load(output_filename)
        os.remove(output_filename)
        os.remove(tmp_fun)
        return arr


def multipoleContracted(
        t1: Tuple, t2: Tuple, rc: Tuple, e: int, f: int, g: int):
    """
    Matrix entry calculation between two Contracted Gaussian functions.
    Equivalent to < t1| t2 >.

    :param t1: tuple containing the cartesian coordinates and primitve gauss
    function of the bra.
    :type t1: (xyz, (Coeff, Expo))
    :param t2: tuple containing the cartesian coordinates and primitve gauss
    function of the ket.
    :type t2: (xyz, (Coeff, Expo))
    :param rc: Cartesian Coordinates where the multipole is centered
    :type rc: Tuple
    :returns: Float
    """
    gs1 = build_primitives_gaussian(t1)
    gs2 = build_primitives_gaussian(t2)

    return sum(sab_multipole(g1, g2, rc, e, f, g) for g1 in gs1 for g2 in gs2)


def calcMatrixEntry(
        rc: Tuple, e: int, f: int, g: int, molecule: List, dictCGFs: Dict,
        indices_cgfs: Matrix, indices_triang: Matrix) -> float:
    """
    Computed each matrix element using an index a tuple containing the
    cartesian coordinates and the primitives gauss functions.

    :param rc: Multipole center
    :type rc: (Float, Float, Float)
    :param xyz_cgfs: List of tuples containing the cartesian coordinates and
    the primitive gauss functions
    :type xyz_cgfs: [(xyz, (Coeff, Expo))]
    :param ixs: Index of the matrix entry to calculate.
    :type ixs: (Int, Int)
    :returns: matrix entry
    """
    # Number of total orbitals
    result = np.empty(indices_triang.shape[0])

    for k, (i, j) in enumerate(indices_triang):
        # Extract contracted and atom indices
        at_i, cgfs_i_idx = indices_cgfs[i]
        at_j, cgfs_j_idx = indices_cgfs[j]

        # Extract atom
        atom_i = molecule[at_i]
        atom_j = molecule[at_j]
        # Extract CGFs
        cgf_i = dictCGFs[atom_i.symbol.lower()][cgfs_i_idx]
        cgf_j = dictCGFs[atom_j.symbol.lower()][cgfs_j_idx]

        # Contracted Gauss functions and nuclear coordinates
        ti = atom_i.xyz, cgf_i
        tj = atom_j.xyz, cgf_j
        xs = multipoleContracted(ti, tj, rc, e, f, g)
        result[k] = xs

    return result


def calcMtxMultipoleP(
        atoms: List, dictCGFs: Dict, runner='multiprocessing', rc=(0, 0, 0), e=0, f=0, g=0):
    """
    Multipole matrix entry calculation between two Contracted Gaussian functions.
    It uses a partial applied function to pass the center of the multipole `rc`
    and the coefficients of the operator x^e y^f z^g.

    :param atoms: Atomic label and cartesian coordinates
    :param cgfsN: Contracted gauss functions normalized, represented as
    a dictionary list of tuples of coefficients and Exponents.
    :param runner: distributed system to compute the elements of the matrix
    :param rc: Multipole center
    :type rc: (Float, Float, Float)
    :params e,f,g: exponents of X, Y, Z in the multipole operator, respectively.
    :returns: Numpy Array representing a flatten triangular matrix.
    """
    curriedFun = partial(calcMatrixEntry, rc, e, f, g)

    return general_multipole_matrix(atoms, dictCGFs, runner=runner, calculator=curriedFun)


# ==================================<>=========================================
orbitalIndexes = {
    ("S", 0): 0, ("S", 1): 0, ("S", 2): 0,
    ("Px", 0): 1, ("Px", 1): 0, ("Px", 2): 0,
    ("Py", 0): 0, ("Py", 1): 1, ("Py", 2): 0,
    ("Pz", 0): 0, ("Pz", 1): 0, ("Pz", 2): 1,
    ("Dxx", 0): 2, ("Dxx", 1): 0, ("Dxx", 2): 0,
    ("Dxy", 0): 1, ("Dxy", 1): 1, ("Dxy", 2): 0,
    ("Dxz", 0): 1, ("Dxz", 1): 0, ("Dxz", 2): 1,
    ("Dyy", 0): 0, ("Dyy", 1): 2, ("Dyy", 2): 0,
    ("Dyz", 0): 0, ("Dyz", 1): 1, ("Dyz", 2): 1,
    ("Dzz", 0): 0, ("Dzz", 1): 0, ("Dzz", 2): 2,
    ("Fxxx", 0): 3, ("Fxxx", 1): 0, ("Fxxx", 2): 0,
    ("Fxxy", 0): 2, ("Fxxy", 1): 1, ("Fxxy", 2): 0,
    ("Fxxz", 0): 2, ("Fxxz", 1): 0, ("Fxxz", 2): 1,
    ("Fxyy", 0): 1, ("Fxyy", 1): 2, ("Fxyy", 2): 0,
    ("Fxyz", 0): 1, ("Fxyz", 1): 1, ("Fxyz", 2): 1,
    ("Fxzz", 0): 1, ("Fxzz", 1): 0, ("Fxzz", 2): 2,
    ("Fyyy", 0): 0, ("Fyyy", 1): 3, ("Fyyy", 2): 0,
    ("Fyyz", 0): 0, ("Fyyz", 1): 2, ("Fyyz", 2): 1,
    ("Fyzz", 0): 0, ("Fyzz", 1): 1, ("Fyzz", 2): 2,
    ("Fzzz", 0): 0, ("Fzzz", 1): 0, ("Fzzz", 2): 3
}


def calcOrbType_Components(l, x):
    """
    Functions related to the orbital momenta indexes.

    :param l: Orbital momentum label
    :type l: String
    :param x: cartesian Component (x, y or z)
    :param x: Int
    :returns: integer representing orbital momentum l.
    """
    return orbitalIndexes[l, x]


def build_primitives_gaussian(t):
    """
    Creates a primitve Gaussian function represented by a tuple containing
    the Cartesian coordinates where it is centered, the spin momentum label
    (S, Px, Py, Pz, etc.) and the Coefficients and exponent of it.
    """
    r, cgf = t
    cs, es = cgf.primitives
    l = cgf.orbType
    return list(map(lambda rs: (r, l, rs), zip(cs, es)))


def calcIndexTriang(n):
    """
    Calculate the indexes of the matrix that is represented as a flatten array.
    """
    flatDim = (n ** 2 + n) // 2
    xss = np.dstack(np.triu_indices(n))
    return np.reshape(xss, (flatDim, 2))


def createTupleXYZ_CGF(atom, cgfs):
    xyz = atom.xyz
    return map(lambda cs: (xyz, cs), cgfs)


def compute_CGFs_indices(mol: List, dictCGFs: Dict) -> Tuple:
    """
    Create a matrix of indices of dimension nOrbs x 2.
    Where the first column contains the index atom and the second
    the index of the CGFs relative to the atom
    """
    # List of the length for each atom in the molecule
    lens = [len(dictCGFs[at.symbol.lower()]) for at in mol]
    nOrbs = sum(lens)

    # Array containing the index of both atoms and CGFs
    indices = np.empty((nOrbs, 2), dtype=np.int32)

    acc = 0
    for i in range(len(mol)):
        nContracted = lens[i]
        slices = acc + nContracted
        # indices of the CGFs
        indices[acc: slices, 1] = np.arange(nContracted)
        # index of the atom
        indices[acc: slices, 0] = i
        acc += nContracted

    return indices, nOrbs


def compute_block_triang_indices(nOrbs: int, ncores: int) -> List:
    """
    Create the list of indices of the triangular matrix to be
    distribute approximately uniform among the available cores.
    """

    # Indices of the triangular matrix
    indices = np.stack(np.triu_indices(nOrbs), axis=1)

    # Number of entries in a triangular matrix
    dim_triang = indices.shape[0]

    # Number of entries to compute for each cores
    chunk = dim_triang // ncores

    # Remaining entries
    rest = dim_triang % ncores

    xs = []
    acc = 0
    for i in range(ncores):
        b = 1 if i < rest else 0
        upper = acc + chunk + b
        xs.append(indices[acc: upper])
        acc = upper

    return xs


def clean(xs):
    """Remove tmp files"""
    for x in xs:
        if os.path.exists(x):
            os.remove(x)
