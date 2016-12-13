__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
# from multipoleObaraSaika import sab_unfolded
from functools import partial
from math import (pi, sqrt)
from multiprocessing import Pool
from multipoleObaraSaika import obaraSaikaMultipole
from .multipoleIntegrals import calcOrbType_Components
from nac.common import AtomXYZ
from nac.integrals.overlapIntegral import sijContracted

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Numpy type hints
Vector = np.ndarray
Matrix = np.ndarray


# =====================================<>======================================
def calculateCoupling3Points(geometries: Tuple, coefficients: Tuple,
                             dictCGFs: Dict, dt: float,
                             trans_mtx: Matrix=None) -> float:
    """
    Calculate the non-adiabatic interaction matrix using 3 geometries,
    the CGFs for the atoms and molecular orbitals coefficients read
    from a HDF5 File.
    :parameter geometries: Tuple of molecular geometries.
    :type      geometries: ([AtomXYZ], [AtomXYZ], [AtomXYZ])
    :parameter coefficients: Tuple of Molecular Orbital coefficients.
    :type      coefficients: (Matrix, Matrix, Matrix)
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],
    AngularMomentum), Primitive = (Coefficient, Exponent)
    :parameter dt: dynamic integration time
    :type      dt: Float (a.u)
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    """
    r0, r1, r2 = geometries
    css0, css1, css2 = coefficients

    # Dictionary containing the number of CGFs per atoms
    cgfs_per_atoms = {s: len(dictCGFs[s]) for s in dictCGFs.keys()}
    # Dimension of the square overlap matrix
    dim = sum(cgfs_per_atoms[at[0]] for at in r0)

    suv_0 = calcOverlapMtx(dictCGFs, dim, r0, r1)
    suv_0_t = np.transpose(suv_0)
    suv_1 = calcOverlapMtx(dictCGFs, dim, r1, r2)
    suv_1_t = np.transpose(suv_1)

    mtx_sji_t0 = calculate_spherical_overlap(suv_0, css0, css1, trans_mtx)
    mtx_sji_t1 = calculate_spherical_overlap(suv_1, css1, css2, trans_mtx)
    mtx_sij_t0 = calculate_spherical_overlap(suv_0_t, css1, css0, trans_mtx)
    mtx_sij_t1 = calculate_spherical_overlap(suv_1_t, css2, css1, trans_mtx)
    cte = 1.0 / (4.0 * dt)

    return cte * (3 * (mtx_sji_t1 - mtx_sij_t1) + (mtx_sij_t0 - mtx_sji_t0))


def calculate_spherical_overlap(suv: Matrix, css0: Matrix,
                                css1: Matrix, trans_mtx: Matrix) -> Matrix:
    """
    Calculate the Overlap Matrix between molecular orbitals at different times.
    """
    css0T = np.transpose(css0)
    if trans_mtx is not None:
        # Overlap in Sphericals
        transpose = np.transpose(trans_mtx)
        suv = np.dot(trans_mtx, np.dot(suv, transpose))

    return np.dot(css0T, np.dot(suv, css1))


def calcOverlapMtx(dictCGFs: Dict, dim: int,
                   mol0: List, mol1: List):
    """
    Parallel calculation of the overlap matrix using the atomic
    basis at two different geometries: R0 and R1.
    """
    fun_overlap = partial(calc_overlap_row, dictCGFs, mol1, dim)
    fun_lookup = partial(lookup_cgf, mol0, dictCGFs)

    with Pool() as p:
        xss = p.map(partial(apply_nested, fun_overlap, fun_lookup),
                    range(dim))

    return np.stack(xss)


def apply_nested(f, g, i):
    return f(*g(i))


def calc_overlap_row(dictCGFs: Dict, mol1: List, dim: int,
                     xyz_atom0: List, cgf_i: Tuple) -> Vector:
    """
    Calculate the k-th row of the overlap integral using
    2 CGFs  and 2 different atomic coordinates.
    """
    row = np.empty(dim)
    acc = 0
    for s, xyz_atom1 in mol1:
        cgfs_j = dictCGFs[s]
        nContracted = len(cgfs_j)
        vs = calc_overlap_atom(xyz_atom0, cgf_i, xyz_atom1, cgfs_j)
        row[acc: acc + nContracted] = vs
        acc += nContracted
    return row


def calc_overlap_atom(xyz_0: List, cgf_i: Tuple, xyz_1: List,
                      cgfs_atom_j: Tuple) -> Vector:
    """
    Compute the overlap between the CGF_i of atom0 and all the
    CGFs of atom1
    """
    # li = cgf_i.orbType
    # ps_i = cgf_i.primitives
    # rs = np.empty(len(cgfs_atom_j))
    # for j, cgf_j in enumerate(cgfs_atom_j):
    #     lj = cgf_j.orbType
    #     ps_j = cgf_j.primitives
    #     rs[j] = apply_contraction(np.array(xyz_0), np.array(xyz_1), li, lj, ps_i, ps_j)

    # return rs
    rs = np.empty(len(cgfs_atom_j))
    for j, cgf_j in enumerate(cgfs_atom_j):
        rs[j] = sijContracted((xyz_0, cgf_i), (xyz_1, cgf_j))

    return rs


def apply_contraction(xyz_0: Vector, xyz_1: Vector,
                      li: str, lj: str,
                      ps_i: Tuple, ps_j: Tuple) -> Vector:
    """
    Calculate the Overlap between two CGFs.
    """
    csi, esi = ps_i
    csj, esj = ps_j
    mtx = np.stack([csi, csj, esi, esj])

    ls_i = np.array(list(map(partial(calcOrbType_Components, li), range(3))))
    ls_j = np.array(list(map(partial(calcOrbType_Components, lj), range(3))))

    fun_sab = partial(sab_unfolded, xyz_0, xyz_1, ls_i, ls_j)
    return np.sum(np.apply_along_axis(lambda xs: fun_sab(*xs), axis=0, arr=mtx))


def lookup_cgf(atoms: List, dictCGFs: Dict, i: int) -> Tuple:
    """
    Search for CGFs number `i` in the dictCGFs.
    """
    if i == 0:
        # return first CGFs for the first atomic symbol
        xyz = atoms[0][1]
        r = dictCGFs[atoms[0].symbol][0]
        return xyz, r
    else:
        acc = 0
        for s, xyz in atoms:
            length = len(dictCGFs[s])
            acc += length
            n = (acc - 1) // i
            if n != 0:
                index = length - (acc - i)
                break

    t = xyz, dictCGFs[s][index]

    return t


def sab_unfolded(r1, r2, ls1, ls2, c1, c2, e1, e2):
    """
    """
    cte = sqrt(pi / (e1 + e2))
    u = e1 * e2 / (e1 + e2)
    p = 1.0 / (2.0 * (e1 + e2))

    rp = (e1 * r1 + e2 * r2) / (e1 + e2)
    rab = r1 - r2
    rpa = rp - r1
    rpb = rp - r2
    s00 = cte * np.exp(-u * rab ** 2.0)
    ps = np.repeat(p, 3)

    arr = np.stack([ps, s00, rpa, rpb, rp, ls1, ls2, np.zeros(3)])

    prod = np.prod(np.apply_along_axis(lambda xs: obaraSaikaMultipole(*xs), 0, arr))

    return c1 * c2 * prod


def coordinates_to_numpy(atom):
    """
    Transform the atomic coordinates to numpy arrays
    """
    return AtomXYZ(atom.symbol, np.array(atom.xyz))
