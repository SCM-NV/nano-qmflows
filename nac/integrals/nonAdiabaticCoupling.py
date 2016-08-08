__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import partial
from multiprocessing import Pool
import numpy as np
import sys
# =============================> Internal modules <============================
from nac.integrals.overlapIntegral import sijContracted
from nac.integrals.multipoleIntegrals import createTupleXYZ_CGF
from qmworks.utils import concat, concatMap
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
    symbols = [x.symbol for x in r0]
    cgfsN = [dictCGFs[s] for s in symbols]

    mtx_sji_t0 = calcuate_Sij(cgfsN, r0, r1, css0, css1, trans_mtx)
    mtx_sij_t0 = calcuate_Sij(cgfsN, r1, r0, css1, css0, trans_mtx)
    mtx_sji_t1 = calcuate_Sij(cgfsN, r1, r2, css1, css2, trans_mtx)
    mtx_sij_t1 = calcuate_Sij(cgfsN, r2, r1, css2, css1, trans_mtx)
    cte = 1.0 / (4.0 * dt)

    return cte * np.add(3 * np.subtract(mtx_sji_t1, mtx_sij_t1),
                        np.subtract(mtx_sij_t0, mtx_sji_t0))


def calcuate_Sij(cgfsN, r0, r1, css0, css1, trans_mtx):
    """
    Calculate the Overlap Matrix between molecular orbitals at different times.
    """
    suv = calcOverlapMtxPar(cgfsN, r0, r1)
    css0T = np.transpose(css0)
    if trans_mtx is not None:
        transpose = np.transpose(trans_mtx)
        suv = np.dot(trans_mtx, np.dot(suv, transpose))  # Overlap in Sphericals

    return np.dot(css0T, np.dot(suv, css1))


def calcOverlapMtxPar(cgfsN, r0, r1):
    """
    Parallel calculation of the overlap matrix using the atomic
    basis at two different geometries: R0 and R1. The rows of the
    matrix are calculated in
    using a pool of processes.
    """
    xyz_cgfs0 = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
                          zip(r0, cgfsN))
    xyz_cgfs1 = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
                          zip(r1, cgfsN))

    dim = len(xyz_cgfs0)
    iss = range(dim)

    with Pool() as p:
        rss = p.map(partial(chunkSij, xyz_cgfs0, xyz_cgfs1, dim), iss)

    result = np.concatenate(rss)

    return result.reshape((dim, dim))


def chunkSij(xyz_cgfs0, xyz_cgfs1, dim, i):
    """
    Calculate the k-th row of the overlap integral using
    the 2 atomics basis at different geometries
    and the total dimension of the overlap matrix
    """
    xs = np.empty(dim)
    for j in range(dim):
        t1 = xyz_cgfs0[i]
        t2 = xyz_cgfs1[j]
        xs[j] = sijContracted(t1, t2)

    return xs


def calcOverlapMtxSeq(cgfsN, r0, r1):
    """
    calculate the overlap matrix using the atomic basis at two
    different geometries: R0 and R1.
    """
    xyz_cgfs0 = np.array(concat([createTupleXYZ_CGF(xs, cgss)
                                 for xs, cgss in zip(r0, cgfsN)]))
    xyz_cgfs1 = np.array(concat([createTupleXYZ_CGF(xs, cgss)
                                 for xs, cgss in zip(r1, cgfsN)]))
    dim = len(xyz_cgfs0)
    xs = np.empty((dim, dim))

    for i in range(dim):
        for j in range(dim):
            t1 = xyz_cgfs0[i]
            t2 = xyz_cgfs1[j]
            xs[i, j] = sijContracted(t1, t2)
            # print(i, j, xs[i, j])
    return xs

