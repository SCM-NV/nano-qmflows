

# ==========> Standard libraries and third-party <===============
from functools import partial
import numpy as np

# ==================> Internal modules <====================
from .overlapIntegral import calcMtxOverlapP
from .multipoleIntegrals import calcMtxMultipoleP

# ==================================<>=========================================
# x,y,z exponents value for the dipole
exponents = [{'e': 1, 'f': 0, 'g': 0}, {'e': 0, 'f': 1, 'g': 0},
             {'e': 0, 'f': 0, 'g': 1}]


def computeIntegralSum(arr, mtx):
    """
    Calculate the operation sum(arr^t mtx arr)
    """
    arrT = np.transpose(arr)

    return np.sum(np.dot(arrT, np.dot(mtx, arr)))


def calculateDipoleCenter(atoms, cgfsN, css, overlap):
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
    mtx_integrals = [calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
                     for kw in exponents]

    xs_sum = list(map(partial(computeIntegralSum, css), mtx_integrals))

    return tuple(map(lambda x: - x / overlap), xs_sum)


def  oscillator_strength(atoms, cgfsN, coeffs, energy):
    """
    :param atoms: Atomic label and cartesian coordinates
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param coeffs: MO coefficients.
    :type coeffs: Numpy array
    :param energy: MO energy.
    :type energy: Double
    """
    sh, = coeffs.shape
    css = np.tile(coeffs, sh)

    overlap = calcMtxOverlapP(atoms, cgfsN)
    overlap_sum = computeIntegralSum(overlap, css)
    rc = calculateDipoleCenter(atoms, cgfsN, css, overlap_sum)

    mtx_integrals = [calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
                     for kw in exponents]

    sum_integrals = sum(lambda x: x ** 2,
                        map(partial(computeIntegralSum, css), mtx_integrals))

    return (2 / 3) * energy * sum_integrals
