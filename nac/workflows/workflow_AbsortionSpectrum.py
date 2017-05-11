import matplotlib
matplotlib.use('Agg')

# ================> Python Standard  and third-party <==========
from functools import partial
from noodles import schedule
from nac.common import (retrieve_hdf5_data, triang2mtx)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.schedule.components import calculate_mos
from qmworks import run

import logging
import matplotlib.pyplot as plt
import numpy as np

# Type hints
from typing import (Dict, List)

# ==============================> Main <==================================
h2ev = 27.2114  # hartrees to electronvolts


def simulate_absoprtion_spectrum(
        package_name: str, project_name: str, package_args: Dict,
        guess_args: Dict=None, geometries: List=None,
        dictCGFs: Dict=None, enumerate_from: int=0,
        initial_states: List=None, final_states: List=None,
        calc_new_wf_guess_on_points: str=None,
        path_hdf5: str=None, package_config: Dict=None,
        traj_folders: List=None, hdf5_trans_mtx: str=None,
        geometry_units='angstrom'):
    """
    Compute the oscillator strength

    :param package_name: Name of the package to run the QM simulations.
    :param project_name: Folder name where the computations
    are going to be stored.
    :param geometry:string containing the molecular geometry.
    :param package_args: Specific settings for the package
    :param guess_args: Specific settings for guess calculate with `package`.
    :type package_args: dict
    :param initial_states: List of the initial Electronic states.
    :type initial_states: [Int]
    :param final_states: List containing the sets of possible electronic
    states.
    :type final_states: [[Int]]
    :param calc_new_wf_guess_on_points: Points where the guess wave functions
    are calculated.
    :param package_config: Parameters required by the Package.
    :returns: None
    """
    # Start logging event
    file_log = '{}.log'.format(project_name)
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(levelname)s:%(message)s  %(asctime)s\n',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config)

    # Schedule the function the compute the Oscillator Strenghts
    scheduleOscillator = schedule(calcOscillatorStrenghts)

    first_geometry = None

    oscillators = scheduleOscillator(
        project_name, mo_paths_hdf5, dictCGFs, first_geometry, path_hdf5,
        hdf5_trans_mtx=hdf5_trans_mtx, initial_states=initial_states,
        final_states=final_states)

    run(oscillators)
    print("Calculation Done")


def calcOscillatorStrenghts(
        project_name: str, mo_paths_hdf5: List, dictCGFs: Dict,
        atoms: List, path_hdf5: str, hdf5_trans_mtx: str=None,
        initial_states: List=None, final_states: List=None):

    """
    Use the Molecular orbital Energies and Coefficients to compute the
    oscillator_strength.

    :param project_name: Folder name where the computations
    are going to be stored.
    :param mo_paths_hdf5: Path to the MO coefficients and energies in the
    HDF5 file.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :param atoms: Molecular geometry.
    :type atoms: [namedtuple("AtomXYZ", ("symbol", "xyz"))]
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param hdf5_trans_mtx: path to the transformation matrix in the HDF5 file.
    :param initial_states: List of the initial Electronic states.
    :type initial_states: [Int]
    :param final_states: List containing the sets of possible electronic
    states.
    :type final_states: [[Int]]
    """
    # Get logger
    logger = logging.getLogger(__name__)

    # Contracted Gaussian functions normalized
    cgfsN = [dictCGFs[x.symbol] for x in atoms]

    es, coeffs = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[0])

    # If the MO orbitals are given in Spherical Coordinates transform then to
    # Cartesian Coordinates.
    if hdf5_trans_mtx is not None:
        trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)

    overlap_CGFS = calcOverlapCGFS(atoms, cgfsN, trans_mtx)

    oscillators = []
    for initialS, fs in zip(initial_states, final_states):
        css_i = coeffs[:, initialS]
        energy_i = es[initialS]
        sum_overlap = np.dot(css_i, np.dot(overlap_CGFS, css_i))
        rc = calculateDipoleCenter(atoms, cgfsN, css_i, trans_mtx, sum_overlap)
        mtx_integrals_spher = calcDipoleCGFS(atoms, cgfsN, rc, trans_mtx)
        logger.info("Dipole center is: {}".format(rc))
        xs = []
        for finalS in fs:
            css_j = coeffs[:, finalS]
            energy_j = es[finalS]
            deltaE = energy_j - energy_i

            print("Calculating Fij between ", initialS, " and ", finalS)
            fij = oscillator_strength(css_i, css_j, deltaE, trans_mtx,
                                      mtx_integrals_spher)
            xs.append(fij)
            with open("oscillator_strengths.out", 'a') as f:
                x = 'transition {:d} -> {:d} f_ij = {:f}\n'.format(initialS,
                                                                   finalS, fij)
                f.write(x)
        oscillators.append(xs)

    return oscillators


def transform2Spherical(trans_mtx, matrix):
    """ Transform from spherical to cartesians"""
    return np.dot(trans_mtx, np.dot(matrix,
                                    np.transpose(trans_mtx)))


def computeIntegralSum(v1, v2, mtx):
    """
    Calculate the operation sum(arr^t mtx arr)
    """
    return np.dot(v1, np.dot(mtx, v2))


def calcOverlapCGFS(atoms, cgfsN, trans_mtx):
    """
    Calculate the Matrix containining the overlap integrals bewtween
    contracted Gauss functions and transform it to spherical coordinates.

    :param atoms: Atomic label and cartesian coordinates in au.
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy Matrix
    """
    _, dimCart = trans_mtx.shape
    # Overlap matrix calculated as a flatten triangular matrix
    overlap_triang = calcMtxOverlapP(atoms, cgfsN)
    # Expand the flatten triangular array to a matrix
    overlap_cart = triang2mtx(overlap_triang, dimCart)

    return transform2Spherical(trans_mtx, overlap_cart)


def calcDipoleCGFS(atoms, cgfsN, rc, trans_mtx):
    """
    Compute the Multipole matrix in cartesian coordinates and
    expand it to a matrix and finally convert it to spherical coordinates.

    :param atoms: Atomic label and cartesian coordinates in au.
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param trans_mtx: Transformation matrix to translate from Cartesian
    to Sphericals.
    :type trans_mtx: Numpy Matrix
    :returns: tuple(<\Psi_i | x | \Psi_j>, <\Psi_i | y | \Psi_j>,
              <\Psi_i | z | \Psi_j>)
    """
    # x,y,z exponents value for the dipole
    exponents = [{'e': 1, 'f': 0, 'g': 0}, {'e': 0, 'f': 1, 'g': 0},
                 {'e': 0, 'f': 0, 'g': 1}]

    dimCart = trans_mtx.shape[1]
    mtx_integrals_triang = tuple(calcMtxMultipoleP(atoms, cgfsN, rc, **kw)
                                 for kw in exponents)
    mtx_integrals_cart = tuple(triang2mtx(xs, dimCart)
                               for xs in mtx_integrals_triang)
    return tuple(transform2Spherical(trans_mtx, x) for x
                 in mtx_integrals_cart)


def calculateDipoleCenter(atoms, cgfsN, css, trans_mtx, overlap):
    """
    Calculate the point where the dipole is centered.

    :param atoms: Atomic label and cartesian coordinates
    type atoms: List of namedTuples
    :param cgfsN: Contracted gauss functions normalized, represented as
    a list of tuples of coefficients and Exponents.
    type cgfsN: [(Coeff, Expo)]
    :param overlap: Integral < \Psi_i | \Psi_i >.
    :type overlap: Float
    To calculate the origin of the dipole we use the following property,

    ..math::
    \braket{\Psi_i \mid \hat{x_0} \mid \Psi_i} =
                       - \braket{\Psi_i \mid \hat{x} \mid \Psi_i}
    """
    rc = (0, 0, 0)

    mtx_integrals_spher = calcDipoleCGFS(atoms, cgfsN, rc, trans_mtx)
    xs_sum = list(map(partial(computeIntegralSum, css, css),
                      mtx_integrals_spher))

    return tuple(map(lambda x: - x / overlap, xs_sum))


def oscillator_strength(css_i, css_j, energy, trans_mtx,
                        mtx_integrals_spher):
    """
    Calculate the oscillator strength between two state i and j using a
    molecular geometry in atomic units, a set of contracted gauss functions
    normalized, the coefficients for both states, the nergy difference between
    the states and a matrix to transform from cartesian to spherical
    coordinates in case the coefficients are given in cartesian coordinates.

    :param css_i: MO coefficients of initial state
    :type coeffs: Numpy Matrix.
    :param css_j: MO coefficients of final state
    :type coeffs: Numpy Matrix.
    :param energy: energy difference i -> j.
    :type energy: Double
    :returns: Oscillator strength (float)
    """
    sum_integrals = sum(x ** 2 for x in
                        map(partial(computeIntegralSum, css_i, css_j),
                            mtx_integrals_spher))

    return (2 / 3) * energy * sum_integrals


def graphicResult(rs, project_name, path_hdf5, mo_paths_hdf5,
                  initial_states=None, final_states=None, deviation=0.1):
    """
    """
    def distribution(x, mu=0, sigma=1):
        """
        Normal Gaussian distribution
        """
        return 1 / (sigma * np.sqrt(2 * np.pi)) * \
            np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def calcDistribution(npoints, mu=0, sigma=1):
        """
        """
        xs = np.linspace(mu - 2, mu + 2, num=npoints)
        ys = np.apply_along_axis(lambda x:
                                 distribution(x, mu, sigma), 0, xs)
        return xs, ys

    oscillators = [[x * h2ev for x in ys] for ys in rs]
    es = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[0][0])

    initialEs = es[initial_states]
    finalEs = [es[v] for v in final_states]
    deltas = [[h2ev * (x - e_i) for x in es_f] for (e_i, es_f) in
              zip(initialEs, finalEs)]

    print('oscillators: ', oscillators)
    print('Energies: ', deltas)
    plt.title('Absorption Spectrum')
    plt.ylabel('Intensity [au]')
    plt.xlabel('Energy [ev]')
    plt.xlim([1, 3.5])
    plt.tick_params(axis='y', which='both', left='off', labelleft='off')
    colors = ['g', 'r', 'b', 'y']
    for k, (es, fs) in enumerate(zip(deltas, oscillators)):
        for e, f in zip(es, fs):
            xs, ys = calcDistribution(1000, mu=e, sigma=deviation)
            plt.plot(xs, ys * f, colors[k])
    plt.savefig('spectrum.pdf', format='pdf')
    plt.show()

