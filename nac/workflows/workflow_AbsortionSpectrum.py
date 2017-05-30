
__all__ = ['workflow_oscillator_strength']

from collections import namedtuple
from itertools import chain
from noodles import (gather, schedule)
from nac.common import (
    Matrix, Vector, change_mol_units, getmass, h2ev,
    retrieve_hdf5_data, triang2mtx)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.schedule.components import calculate_mos
from nac.schedule.scheduleCoupling import (
    calculate_overlap, compute_the_fixed_phase_overlaps)
from qmworks import run
from qmworks.parsers import parse_string_xyz
from scipy import sparse

import logging
import numpy as np

# Type hints
from typing import (Dict, List, Tuple)

# Get logger
logger = logging.getLogger(__name__)

# Named tupel
OScillator = namedtuple("Oscillator", "FIXME")

# ==============================> Main <==================================


def workflow_oscillator_strength(
        package_name: str, project_name: str, package_args: Dict,
        guess_args: Dict=None, geometries: List=None,
        dictCGFs: Dict=None, enumerate_from: int=0,
        calc_new_wf_guess_on_points: str=None,
        path_hdf5: str=None, package_config: Dict=None,
        work_dir: str=None,
        initial_states: List=None, final_states: List=None,
        traj_folders: List=None, hdf5_trans_mtx: str=None,
        nHOMO: int=None, couplings_range: Tuple=None,
        calculate_oscillator_every: int=50,
        convolution='gaussian',
        geometry_units='angstrom', **kwargs):
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

    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(
        package_name, geometries, project_name, path_hdf5, traj_folders,
        package_args, guess_args, calc_new_wf_guess_on_points,
        enumerate_from, package_config=package_config)

    # Overlap matrix at two different times
    if len(geometries) > 1:
        promised_overlaps = calculate_overlap(
            project_name, path_hdf5, dictCGFs, geometries, mo_paths_hdf5,
            hdf5_trans_mtx, enumerate_from, nHOMO=nHOMO,
            couplings_range=couplings_range)

        # track the orbitals duringh the MD
        schedule_compute_swaps = schedule(compute_swapped_indexes)

        swaps = schedule_compute_swaps(
            promised_overlaps, path_hdf5, project_name, enumerate_from, nHOMO)
    else:
        # Use only the first point to compute the oscillator strength
        dim = sum(couplings_range)
        swaps = np.arange(dim).reshape(1, dim)

    # geometries in atomic units
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in geometries]

    # Contracted Gaussian functions normalized
    cgfsN = [dictCGFs[x.symbol] for x in molecules_au[0]]

    # Schedule the function the compute the Oscillator Strenghts
    scheduleOscillator = schedule(calcOscillatorStrenghts)

    oscillators = gather(
        *[scheduleOscillator(
            i, swaps, project_name, mo_paths_hdf5, cgfsN, mol,
            path_hdf5, hdf5_trans_mtx=hdf5_trans_mtx,
            initial_states=initial_states, final_states=final_states)
          for i, mol in enumerate(molecules_au)
          if i % calculate_oscillator_every == 0])

    if len(geometries) > 1:
        # Compute the cross section
        schedule_cross_section = schedule(compute_cross_section_function)

        promised_cross_section = schedule_cross_section(
            oscillators, path_hdf5, mo_paths_hdf5, convolution,
            calculate_oscillator_every)

        data = run(
            gather(oscillators), folder=work_dir)
        
        # data = run(
        #     gather(promised_cross_section, oscillators), folder=work_dir)
    else:
        data = run(oscillators, folder=work_dir)
        # cross_section = None

    for xs in list(chain(*data)):
        for args in xs:
            write_information(*args)

    print("Calculation Done")

    return data


def compute_cross_section_function(
        oscillators: List, path_hdf5: str, mo_paths_hdf5: List,
        convolution: str, calculate_oscillator_every: int) -> float:
    """
    Compute the photoabsorption cross section as a function of the energy.
    See: The UV absorption of nucleobases: semi-classical ab initio spectra
    simulations. Phys. Chem. Chem. Phys., 2010, 12, 4959–4967
    """
    pass
    # # speed of light in a.u.
    # c = 137
    # # Constant
    # cte = 2 * (np.pi ** 2) / c

    # # rearrange oscillator strengths by initial states
    # def fun_cross_section(energy: float) -> float:
    #     return cte * sum(sum(sum() / len(ws) for ws in zip(*arr))
    #                      for arr in zip(*oscillators))


def gaussian_distribution(center, delta: float, size: int=1) -> Vector:
    """
    Return gaussian as described at:
    Phys. Chem. Chem. Phys., 2010, 12, 4959–4967
    """
    return np.random.normal(loc=center, scale=delta / 2, size=size)


def lorentzian_distribution(center: float, parameter: float, size: int=1) -> Vector:
    """
    Return a Lorentzian as described at:
    Phys. Chem. Chem. Phys., 2010, 12, 4959–4967
    """
    pass
    # c = delta / 2 * np.pi


def compute_swapped_indexes(promised_overlaps, path_hdf5, project_name,
                            enumerate_from, nHOMO):
    """
    Track the swap between the Molecular orbitals during the
    Molecular dynamics
    """
    logger.info("Tracking the swaps between Molecular orbitals")
    # Overlaps and swaps
    fixed_overlaps_and_swaps = compute_the_fixed_phase_overlaps(
        promised_overlaps, path_hdf5, project_name, enumerate_from, nHOMO)

    swaps = fixed_overlaps_and_swaps[1]
    dim = swaps.shape[0]

    # Accumulate the swaps
    for i in range(1, dim):
        swaps[i] = swaps[i, swaps[i - 1]]

    return swaps


def calcOscillatorStrenghts(
        i: int, swaps: Matrix, project_name: str,
        mo_paths_hdf5: str, cgfsN: List,
        atoms: List, path_hdf5: str, hdf5_trans_mtx: str=None,
        initial_states: List=None, final_states: List=None):

    """
    Use the Molecular orbital Energies and Coefficients to compute the
    oscillator_strength.

    :param i: time frame
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
    # Energy and coefficients at time t
    es, coeffs = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[i])

    # Apply the swap that took place during the MD
    swapped_initial_states = swaps[i, initial_states]
    swapped_final_states = swaps[i, final_states]

    # If the MO orbitals are given in Spherical Coordinates transform then to
    # Cartesian Coordinates.
    if hdf5_trans_mtx is not None:
        trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)

    logger.info("Computing the oscillator strength at time: {}".format(i))
    # Overlap matrix

    # Origin of the dipole
    rc = compute_center_of_mass(atoms)

    # Dipole matrix element in spherical coordinates
    mtx_integrals_spher = calcDipoleCGFS(atoms, cgfsN, rc, trans_mtx)

    oscillators = [
        compute_oscillator_strength(
            rc, atoms, cgfsN, es, coeffs, mtx_integrals_spher, initialS, fs)
        for initialS, fs in zip(swapped_initial_states, swapped_final_states)]

    return oscillators


def compute_oscillator_strength(
        rc: Tuple, atoms: List, cgfsN: List, es: Vector, coeffs: Matrix,
        mtx_integrals_spher: Matrix, initialS: int, fs: List):
    """
    Compute the oscillator strenght using the matrix elements of the position
    operator:

    .. math:
    f_i->j = 2/3 * E_i->j * ∑^3_u=1 [ <ψi | r_u | ψj> ]^2

    where Ei→j is the single particle energy difference of the transition
    from the Kohn-Sham state ψi to state ψj and rμ = x,y,z is the position
    operator.
    """
    # Retrieve the molecular orbital coefficients and energies
    css_i = coeffs[:, initialS]
    energy_i = es[initialS]

    # Compute the oscillator strength
    xs = []
    for finalS in fs:
        # Get the molecular orbitals coefficients and energies
        css_j = coeffs[:, finalS]
        energy_j = es[finalS]
        deltaE = energy_j - energy_i

        # compute the oscillator strength and the transition dipole components
        fij, components = oscillator_strength(
            css_i, css_j, deltaE, mtx_integrals_spher)

        st = 'transition {:d} -> {:d} Fij = {:f}\n'.format(
            initialS, finalS, fij)
        logger.info(st)
        xs.append((initialS, finalS, deltaE, fij, components))

    return xs


def write_information(initialS: int, finalS: int, deltaE: float, fij: float,
                      components: Tuple) -> None:
    """
    Write oscillator strenght information in one file
    """
    energy = deltaE * h2ev
    fmt = '{}_{} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}\n'.format(
        initialS, finalS, energy, fij, *components)

    with open('oscillators.txt', 'a') as f:
        f.write(fmt)


def transform2Spherical(trans_mtx: Matrix, matrix: Matrix) -> Matrix:
    """
    Transform from spherical to cartesians using the sparse representation
    """
    trans_mtx = sparse.csr_matrix(trans_mtx)
    transpose = trans_mtx.transpose()

    return trans_mtx.dot(sparse.csr_matrix.dot(matrix, transpose))


def calcOverlapCGFS(
        atoms: List, cgfsN: List, trans_mtx: Matrix) -> Matrix:
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
    dimCart = trans_mtx.shape[1]
    # Overlap matrix calculated as a flatten triangular matrix
    overlap_triang = calcMtxOverlapP(atoms, cgfsN)
    # Expand the flatten triangular array to a matrix
    overlap_cart = triang2mtx(overlap_triang, dimCart)

    return transform2Spherical(trans_mtx, overlap_cart)


def calcDipoleCGFS(
        atoms: List, cgfsN: List, rc: Tuple, trans_mtx: Matrix) -> Matrix:
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
    :returns: tuple(<ψi | x | ψj>, <ψi | y | ψj>, <ψi | z | ψj> )
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


def oscillator_strength(css_i: Matrix, css_j: Matrix, energy: float,
                        mtx_integrals_spher: Matrix) -> Tuple:
    """
    Calculate the oscillator strength between two state i and j using a
    molecular geometry in atomic units, a set of contracted gauss functions
    normalized, the coefficients for both states, the nergy difference between
    the states and a matrix to transform from cartesian to spherical
    coordinates in case the coefficients are given in cartesian coordinates.

    :param css_i: MO coefficients of initial state
    :param css_j: MO coefficients of final state
    :param energy: energy difference i -> j.
    :returns: Oscillator strength
    """
    components = tuple(
        map(lambda mtx: np.dot(css_i, np.dot(mtx, css_j)),
            mtx_integrals_spher))

    sum_integrals = sum(x ** 2 for x in components)

    fij = (2 / 3) * energy * sum_integrals

    return fij, components


def compute_center_of_mass(atoms: List) -> Tuple:
    """
    Compute the center of mass of a molecule
    """
    # Get the masses of the atoms
    symbols = map(lambda at: at.symbol, atoms)
    masses = np.array([getmass(s) for s in symbols])
    total_mass = np.sum(masses)

    # Multiple the mass by the coordinates
    mrs = [getmass(at.symbol) * np.array(at.xyz) for at in atoms]
    xs = np.sum(mrs, axis=0)

    # Center of mass
    cm = xs / total_mass

    return tuple(cm)
