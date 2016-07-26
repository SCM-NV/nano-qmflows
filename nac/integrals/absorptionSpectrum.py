
__all__ = ['oscillator_strength', 'calculateDipoleCenter']

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


def computeIntegralSum(arrT, arr, mtx):
    """
    Calculate the operation sum(arr^t mtx arr)
    """
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

    cssT = np.transpose(css)
    xs_sum = list(map(partial(computeIntegralSum, cssT, css), mtx_integrals))

    return tuple(map(lambda x: - x / overlap), xs_sum)


def  oscillator_strength(atoms, cgfsN, css_i, css_j, energy):
    """
    :param atoms: Atomic label and cartesian coordinates
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param css_i: MO coefficients of initial state
    :type coeffs: Numpy Matrix.
    :param css_j: MO coefficients of final state
    :type coeffs: Numpy Matrix.
    :param energy: MO energy.
    :type energy: Double
    """
    overlap = calcMtxOverlapP(atoms, cgfsN)
    css_i_T = np.transpose(css_i)
    overlap_sum = computeIntegralSum(css_i_T, css_i, overlap)
    rc = calculateDipoleCenter(atoms, cgfsN, css_i, overlap_sum)

    mtx_integrals = [calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
                     for kw in exponents]

    sum_integrals = sum(lambda x: x ** 2,
                        map(partial(computeIntegralSum, css_i_T, css_j),
                            mtx_integrals))

    return (2 / 3) * energy * sum_integrals
