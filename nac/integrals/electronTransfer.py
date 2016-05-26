__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from functools import partial
from itertools import starmap
from multiprocessing import Pool
import numpy as np
import sys
# ==================> Internal modules <==========
from nac.integrals.overlapIntegral import calcMtxOverlapP

# ==================================<>=========================================
au_time = 2.41888432e-2  # 1 au of time is  2.41888432e-2 femtoseconds


def photoExcitationRate(geometries, dictCGFs, tuple_time_coefficients,
                        tuple_mos_coefficients, trans_mtx=None, dt=1):
    """
    Calculate the Electron transfer rate, using both adiabatic and nonadiabatic
    components, using equation number 8 from:
    J. AM. CHEM. SOC. 2005, 127, 7941-7951. Ab Initio Nonadiabatic Molecular
    Dynamics of the Ultrafast Electron Injection acrossthe Alizarin-TiO2
    Interface.

    :param geometry: Molecular geometry
    :type geometry: List of Atom objects
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF], CGF = ([Primitives],AngularMomentum),
    Primitive = (Coefficient, Exponent)
    :param tuple_time_coefficients: Time-dependent coefficients
    at time t - dt, t and t + dt.
    :type tuple_time_coefficients: Tuple of Numpy Arrays
    :param tuple_mos_coefficients: Tuple of Molecular orbitals at
    time t - dt, t and t + dt.
    :type tuple_mos_coefficients: Tuple of Numpy Arrays
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy matrix
    :returns: tuple containing both nonadiabatic and adiabatic components
    """
    # transform time to atomic units
    dt = dt / au_time
    # Geometry at time t
    r1 = geometries[1]
    symbols = [x.symbol for x in r1]
    cgfsN = [dictCGFs[s] for s in symbols]

    # Overlap matrices
    s0, s1, s2 = tuple(starmap(partial(overlap_molecular, cgfsN, trans_mtx),
                               zip(geometries, tuple_mos_coefficients)))
    # NonAdiabatic component
    nonadiabatic = electronTransferNA(r1, tuple_time_coefficients, s1)
    # Adiabatic component
    css1 = tuple_time_coefficients[1]
    adiabatic = electronTransferAdiabatic(css1, np.stack([s0, s1, s2]))

    return nonadiabatic, adiabatic


def electronTransferNA(geometry, tuple_coefficients, overlap, dt=1):
    """
    Calculate the Nonadiabatic component of an Electron transfer process.

    :param geometry: Molecular geometry
    :type geometry: List of Atom objects
    :param tuple_coefficients: Time-dependent coefficients at time
    t - dt, t and t + dt.
    :type tuple_coefficients: Tuple of Numpy Arrays
    :param overlap: Overlap matrix
    :type overlap: Numpy Martrix
    :param dt: Integration time.
    :type dt: Float
    """
    dim = tuple_coefficients[0].shape
    xs = [x.reshape(1, dim) for x in tuple_coefficients]
    xss = np.stack([x * np.transpose(x) for x in xs])
    # Use a second order derivative for the time dependent coefficients
    derivatives = np.apply_along_axis(threePointDerivative, 0, xss)

    return np.sum(derivatives * overlap)


def electronTransferAdiabatic(coefficients, tensor_overlap):
    """
    Calculate the Adiabatic component of an Electron transfer process.
    :param tuple_coefficients: Time-dependent coefficients at time t.
    :type tuple_coefficients: Numpy Array
    :param tensor_overlap: Tensor containing the overlap matrices at time
    t - dt, t and t + dt.
    :type tensor_overlap: Numpy 3-D array

    """
    dim = coefficients.shape
    coefficients = coefficients.reshape(1, dim)
    css = coefficients * np.transpose(coefficients)

    overlap_derv = np.apply_along_axis(threePointDerivative, 0, tensor_overlap)

    return np.sum(css * overlap_derv)


def overlap_molecular(cgfsN, trans_mtx, geometry, mos):
    """
    Calculate the Overlap matrix using the given geometry and Molecular
    orbitals.

    :param geometry: Molecular geometry
    :type geometry: List of Atom objects
    :param cgfsN: List of Contracted Gaussian functions
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy matrix
    :param mos: Molecular orbitals coefficients
    :type mos: Numpy Martrix
    """
    atomic_overlap = calcMtxOverlapP(geometry, cgfsN)
    mosT = np.transpose(mos)
    if trans_mtx is not None:
        transpose = np.transpose(trans_mtx)
        # Overlap in Sphericals
        suv = np.dot(trans_mtx, np.dot(atomic_overlap, transpose))

    return np.dot(mosT, np.dot(suv, mos))


def threePointDerivative(arr, dt=1):
    """Calculate the numerical derivatives using a Two points formula"""

    return (arr[0] - arr[2]) / (2 * dt)

# def calcuate_Sij(cgfsN, r0, r1, css0, css1, trans_mtx):
#     """
#     Calculate the Overlap Matrix between molecular orbitals at different times.
#     """
#     suv = calcOverlapMtxPar(cgfsN, r0, r1)
#     css0T = np.transpose(css0)
#     if trans_mtx is not None:
#         transpose = np.transpose(trans_mtx)
#         suv = np.dot(trans_mtx, np.dot(suv, transpose))  # Overlap in Sphericals

#     return np.dot(css0T, np.dot(suv, css1))


# def calcOverlapMtxPar(cgfsN, r0, r1):
#     """
#     Parallel calculation of the the overlap matrix using the atomic
#     basis at two different geometries: R0 and R1. The rows of the
#     matrix are calculated in
#     using a pool of processes.
#     """
#     xyz_cgfs0 = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
#                           zip(r0, cgfsN))
#     xyz_cgfs1 = concatMap(lambda rs: createTupleXYZ_CGF(*rs),
#                           zip(r1, cgfsN))

#     dim = len(xyz_cgfs0)
#     iss = range(dim)

#     pool = Pool()
#     rss = pool.map(partial(chunkSij, xyz_cgfs0, xyz_cgfs1, dim), iss)
#     pool.close()
#     result = np.concatenate(rss)

#     return result.reshape((dim, dim))


# def chunkSij(xyz_cgfs0, xyz_cgfs1, dim, i):
#     """
#     Calculate the k-th row of the overlap integral using
#     the 2 atomics basis at different geometries
#     and the total dimension of the overlap matrix
#     """
#     xs = np.empty(dim)
#     for j in range(dim):
#         t1 = xyz_cgfs0[i]
#         t2 = xyz_cgfs1[j]
#         xs[j] = sijContracted(t1, t2)

#     return xs


# def calcOverlapMtxSeq(cgfsN, r0, r1):
#     """
#     calculate the overlap matrix using the atomic basis at two
#     different geometries: R0 and R1.
#     """
#     xyz_cgfs0 = np.array(concat([createTupleXYZ_CGF(xs, cgss)
#                                  for xs, cgss in zip(r0, cgfsN)]))
#     xyz_cgfs1 = np.array(concat([createTupleXYZ_CGF(xs, cgss)
#                                  for xs, cgss in zip(r1, cgfsN)]))
#     dim = len(xyz_cgfs0)
#     xs = np.empty((dim, dim))

#     for i in range(dim):
#         for j in range(dim):
#             t1 = xyz_cgfs0[i]
#             t2 = xyz_cgfs1[j]
#             xs[i, j] = sijContracted(t1, t2)
#             # print(i, j, xs[i, j])
#     return xs

