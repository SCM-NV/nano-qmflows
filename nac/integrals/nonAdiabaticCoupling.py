__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import partial
from multiprocessing import Pool
from nac.integrals.overlapIntegral import sijContracted

import numpy as np

# =====================================<>======================================


def calculateCoupling3Points(geometries, coefficients, dictCGFs, dt,
                             trans_mtx=None):
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
    cgfs_per_atoms = {s: len(extract_labels(dictCGFs[s]))
                      for s in dictCGFs.keys()}
    # Dimension of the square overlap matrix
    dim = sum(cgfs_per_atoms[at[0]] for at in r0)

    suv_0 = calcOverlapMtx(dictCGFs, cgfs_per_atoms, dim, r0, r1)
    suv_0_t = np.transpose(suv_0)
    suv_1 = calcOverlapMtx(dictCGFs, cgfs_per_atoms, dim, r1, r2)
    suv_1_t = np.transpose(suv_1)

    mtx_sji_t0 = calculate_spherical_overlap(suv_0, css0, css1, trans_mtx)
    mtx_sji_t1 = calculate_spherical_overlap(suv_1, css1, css2, trans_mtx)
    mtx_sij_t0 = calculate_spherical_overlap(suv_0_t, css1, css0, trans_mtx)
    mtx_sij_t1 = calculate_spherical_overlap(suv_1_t, css2, css1, trans_mtx)
    cte = 1.0 / (4.0 * dt)

    return cte * (3 * (mtx_sji_t1 - mtx_sij_t1) + (mtx_sij_t0 - mtx_sji_t0))


def calculate_spherical_overlap(suv, css0, css1, trans_mtx):
    """
    Calculate the Overlap Matrix between molecular orbitals at different times.
    """
    css0T = np.transpose(css0)
    if trans_mtx is not None:
        # Overlap in Sphericals
        transpose = np.transpose(trans_mtx)
        suv = np.dot(trans_mtx, np.dot(suv, transpose))

    return np.dot(css0T, np.dot(suv, css1))


def calcOverlapMtx(dictCGFs, cgfs_per_atoms, dim, r0, r1):
    """
    Parallel calculation of the overlap matrix using the atomic
    basis at two different geometries: R0 and R1.
    """
    fun_overlap = partial(calc_overlap_row, dictCGFs, r1, dim)
    fun_lookup = partial(lookup_cgf, r0, cgfs_per_atoms, dictCGFs)

    with Pool() as p:
        xss = p.map(partial(apply_nested, fun_overlap, fun_lookup),
                    range(dim))

    return np.stack(xss)


def apply_nested(f, g, i):
    return f(*g(i))


def calc_overlap_row(dictCGFs, r1, dim, xyz_atom0, cgf_i):
    """
    Calculate the k-th row of the overlap integral using
    2 CGFs  and 2 different atomic coordinates.
    """
    row = np.empty(dim)
    acc = 0
    for s, xyz_atom1 in r1:
        cgfs_j = dictCGFs[s]
        nContracted = len(cgfs_j)
        vs = calc_overlap_atom(xyz_atom0, cgf_i, xyz_atom1, cgfs_j)
        row[acc: acc + nContracted] = vs
        acc += nContracted
    return row


def calc_overlap_atom(xyz_0, fi, xyz_1, cgfs_j):
    """
    Compute the overlap between the CGF_i of atom0 and all the
    CGFs of atom1
    """
    rs = np.empty(len(cgfs_j))
    for j, fj in enumerate(cgfs_j):
        rs[j] = sijContracted((xyz_0, fi), (xyz_1, fj))

    return rs


def lookup_cgf(atoms, cgfs_per_atoms, dictCGFs, i):
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
            length = cgfs_per_atoms[s]
            acc += length
            n = (acc - 1) // i
            if n != 0:
                index = length - (acc - i)
                break

    t = xyz, dictCGFs[s][index]

    return t


def extract_labels(cgfs):
    """
    Get all the orbital labels from a list of CGFs
    """
    rs = list(map(list, zip(*cgfs)))
    return rs[1]
