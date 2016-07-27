
__all__ = ['oscillator_strength', 'calculateDipoleCenter']

# ==========> Standard libraries and third-party <===============
from functools import partial
import numpy as np

# ==================> Internal modules <====================
from .overlapIntegral import calcMtxOverlapP
from .multipoleIntegrals import calcMtxMultipoleP
from nac.common import triang2mtx
# ==================================<>=========================================
# x,y,z exponents value for the dipole
exponents = [{'e': 1, 'f': 0, 'g': 0}, {'e': 0, 'f': 1, 'g': 0},
             {'e': 0, 'f': 0, 'g': 1}]


def transform2Spherical(mtx, trans_mtx):
    """
    Transform a matrix containing integrals in cartesian coordinates to a matrix
    in spherical coordinates.
    """
    trr = np.transpose(trans_mtx)

    return np.dot(trans_mtx, np.dot(mtx, trr))


def computeIntegralSum(v1, v2, mtx):
    """
    Calculate the operation sum(arr^t mtx arr)
    """
    return np.dot(v1, np.dot(mtx, v2))


def flattenCartesian2MtxSpherical(atoms, cgfsN, rc, trans_mtx):
    """
    Compute the Multipole matrix in cartesian coordinates and
    expand it to a matrix and finally convert it to spherical coordinates.
    """
    dimSpher, dimCart = trans_mtx.shape
    mtx_integrals_triang = tuple(calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
                                 for kw in exponents)
    mtx_integrals_cart = tuple(triang2mtx(xs, dimCart)
                               for xs in mtx_integrals_triang)
    return tuple(transform2Spherical(x, trans_mtx) for x
                 in mtx_integrals_cart)


def calculateDipoleCenter(atoms, cgfsN, css, trans_mtx, overlap):
    """
    Calculate the point where the dipole is centered.
    :param atoms: Atomic label and cartesian coordinates
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]

    To calculate the origin of the dipole we use the following property,

    ..math::
    \braket{\Psi_i \mid \hat{x_0} \mid \Psi_i} =
                       - \braket{\Psi_i \mid \hat{x} \mid \Psi_i}
    """
    rc = (0, 0, 0)

    mtx_integrals_spher = flattenCartesian2MtxSpherical(atoms, cgfsN, rc,
                                                        trans_mtx)
    xs_sum = list(map(partial(computeIntegralSum, css, css),
                      mtx_integrals_spher))

    return tuple(map(lambda x: - x / overlap, xs_sum))


def  oscillator_strength(atoms, cgfsN, css_i, css_j, energy, trans_mtx,
                         overlap):
    """
    Calculate the oscillator strength between two state i and j using a
    molecular geometry in atomic units, a set of contracted gauss functions
    normalized, the coefficients for both states, the nergy difference between
    the states and a matrix to transform from cartesian to spherical
    coordinates in case the coefficients are given in cartesian coordinates.
  
    :param atoms: Atomic label and cartesian coordinates in au.
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param css_i: MO coefficients of initial state
    :type coeffs: Numpy Matrix.
    :param css_j: MO coefficients of final state
    :type coeffs: Numpy Matrix.
    :param energy: energy difference i -> j.
    :type energy: Double
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy Matrix
    :returns: Oscillator strength (float)
    """
    dimSpher, dimCart = trans_mtx.shape
    # Dipole center
    rc = calculateDipoleCenter(atoms, cgfsN, css_i, trans_mtx, overlap)

    print("Dipole center is: ", rc)
    mtx_integrals_spher = flattenCartesian2MtxSpherical(atoms, cgfsN, rc,
                                                        trans_mtx)
    sum_integrals = sum(x ** 2 for x in
                        map(partial(computeIntegralSum, css_i, css_j),
                            mtx_integrals_spher))

    return (2 / 3) * energy * sum_integrals

