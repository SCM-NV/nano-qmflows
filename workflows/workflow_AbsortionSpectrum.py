
# ================> Python Standard  and third-party <==========
from functools import partial
from itertools import tee
from noodles import schedule
from os.path import join
from qmworks import (run, Settings)
from qmworks.parsers import parse_string_xyz

import getpass
import matplotlib.pyplot as plt
import numpy as np
import os
import plams
import shutil

# =========================> Internal modules <================================
from nac.common import (change_mol_units, retrieve_hdf5_data, triang2mtx)
from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.schedule.components import (calculate_mos, create_dict_CGFs,
                                     create_point_folder,
                                     split_file_geometries)
from nac.schedule.scheduleCoupling import schedule_transf_matrix

# ==============================> Main <==================================
h2ev = 27.2114  # hartrees to electronvolts


def simulate_absoprtion_spectrum(package_name, project_name, geometry,
                                 package_args, guess_args=None,
                                 initial_states=None, final_states=None,
                                 calc_new_wf_guess_on_points=[0],
                                 path_hdf5=None, package_config=None,
                                 geometry_units='angstrom'):
    """
    Compute the oscillator strength

    :param package_name: Name of the package to run the QM simulations.
    :type  package_name: String
    :param project_name: Folder name where the computations
    are going to be stored.
    :type project_name: String
    :param geometry:string containing the molecular geometry.
    :type geometry: String
    :param package_args: Specific settings for the package
    :type package_args: dict
    :param package_args: Specific settings for guess calculate with `package`.
    :type package_args: dict
    :param initial_states: List of the initial Electronic states.
    :type initial_states: [Int]
    :param final_states: List containing the sets of possible electronic
    states.
    :type final_states: [[Int]]
    :param calc_new_wf_guess_on_points: Points where the guess wave functions
    are calculated.
    :type use_wf_guess_each: [Int]
    :param package_config: Parameters required by the Package.
    :type package_config: Dict
    :returns: None
    """
    #  Environmental Variables
    cwd = os.path.realpath(".")
    
    basisName = package_args.basis
    work_dir = os.path.join(cwd, project_name)
    if path_hdf5 is None:
        path_hdf5 = os.path.join(work_dir, "quantum.hdf5")

    # Create Work_dir if it does not exist
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)

    # Generate a list of tuples containing the atomic label
    # and the coordinates to generate
    # the primitive CGFs
    atoms = parse_string_xyz(geometry[0])
    if 'angstrom' in geometry_units.lower():
        atoms = change_mol_units(atoms)

    dictCGFs = create_dict_CGFs(path_hdf5, basisName, atoms, package_config)

    # Calculcate the matrix to transform from cartesian to spherical
    # representation of the overlap matrix
    hdf5_trans_mtx = schedule_transf_matrix(path_hdf5, atoms,
                                            basisName, project_name,
                                            packageName=package_name)

    # Create a folder for each point the the dynamics
    traj_folders = create_point_folder(work_dir, 1, 0)

    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, geometry, project_name,
                                  path_hdf5, traj_folders, package_args,
                                  guess_args, calc_new_wf_guess_on_points=[0],
                                  enumerate_from=0,
                                  package_config=package_config)

    scheduleOscillator = schedule(calcOscillatorStrenghts)

    oscillators = scheduleOscillator(project_name, mo_paths_hdf5, dictCGFs,
                                     atoms, path_hdf5,
                                     hdf5_trans_mtx=hdf5_trans_mtx,
                                     initial_states=initial_states,
                                     final_states=final_states)

    scheduleGraphs = schedule(graphicResult)
    
    run(scheduleGraphs(oscillators, project_name, path_hdf5, mo_paths_hdf5,
                       initial_states=initial_states, final_states=final_states))
    print("Calculation Done")


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

    def calcDistribution(x, npoints, mu=0, sigma=1):
        """
        """
        xs = np.linespace(x - 2, x + 2, num=npoints)
        ys = np.apply_apply_along_axis(lambda x: distribution(x, mu, sigma), xs)

        return xs, ys

    xs = map(lambda : x * h2ev, rs)
    es = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[0][0])

    initialEs = es[initial_states]
    finalEs = [es[v] for v in final_states]

    deltas = map(lambda e_i, es_f: [h2ev * (x - e_i) for x in es_f],
                 zip(initialEs, finalEs))

    magnifying_factor = 1
    cm2inch = 0.393700787
    dim1 = 8.25 * cm2inch * magnifying_factor
    dim2 = 6 * cm2inch * magnifying_factor
    plt.figure(figsize=(dim1, dim2), dpi=300 / magnifying_factor)
    plt.title('Absorption Spectrum')
    plt.ylabel('Intensity [au]')
    plt.xlabel('Energy [ev]')
    colors = ['g-' ,'r-', 'b-', 'y-']
    for k, es in enumerate(deltas):
        
        xs, ys = calcDistribution(e_i, 1000, mu=)
        plt.plot(xs, ys, colors[k])
        
    plt.savefig('spectrum.pdf', dpi=300 / magnifying_factor, format='pdf')
    plt.show()


def calcOscillatorStrenghts(project_name, mo_paths_hdf5, dictCGFs, atoms,
                            path_hdf5, hdf5_trans_mtx=None,
                            initial_states=None, final_states=None):

    """
    Use the Molecular orbital Energies and Coefficients to compute the
    oscillator_strength.

    :param project_name: Folder name where the computations
    are going to be stored.
    :type project_name: String
    :param mo_paths_hdf5: Path to the MO coefficients and energies in the
    HDF5 file.
    :paramter dictCGFS: Dictionary from Atomic Label to basis set.
    :type     dictCGFS: Dict String [CGF],
              CGF = ([Primitives], AngularMomentum),
              Primitive = (Coefficient, Exponent)
    :type mo_paths: [String]
    :param atoms: Molecular geometry.
    :type atoms: [namedtuple("AtomXYZ", ("symbol", "xyz"))]
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :type path_hdf5: String
    :param hdf5_trans_mtx: path to the transformation matrix in the HDF5 file.
    :type hdf5_trans_mtx: String
    :param initial_states: List of the initial Electronic states.
    :type initial_states: [Int]
    :param final_states: List containing the sets of possible electronic
    states.
    :type final_states: [[Int]]
    """
    cgfsN = [dictCGFs[x.symbol] for x in atoms]

    es, coeffs = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[0])

    # If the MO orbitals are given in Spherical Coordinates transform then to
    # Cartesian Coordinates.
    trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx) if hdf5_trans_mtx else None

    overlap_CGFS = calcOverlapCGFS(atoms, cgfsN, trans_mtx)

    oscillators = []
    for initialS, fs in zip(initial_states, final_states):
        css_i = coeffs[:, initialS]
        energy_i = es[initialS]
        sum_overlap = np.dot(css_i, np.dot(overlap_CGFS, css_i))
        rc = calculateDipoleCenter(atoms, cgfsN, css_i, trans_mtx, sum_overlap)
        print("Dipole center is: ", rc)
        xs = []
        for finalS in fs:
            mtx_integrals_spher = calcDipoleCGFS(atoms, cgfsN, rc, trans_mtx)
            css_j = coeffs[:, finalS]
            energy_j = es[finalS]
            deltaE = energy_j - energy_i
            print("Calculating Fij between ", initialS, " and ", finalS)
            fij = oscillator_strength(css_i, css_j, deltaE, trans_mtx,
                                      mtx_integrals_spher)
            xs.append(fij)
            with open("oscillator_strengths.out", 'a') as f:
                x = 'transition {:d} -> {:d} f_ij = {:f}\n'.format(initialS, finalS, fij)
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
    dimSpher, dimCart = trans_mtx.shape
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
    dimSpher, dimCart = trans_mtx.shape
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


def  oscillator_strength(css_i, css_j, energy, trans_mtx,
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

# ===================================<>========================================


def main():
    """
    Initialize the arguments to compute the nonadiabatic coupling matrix for
    a given MD trajectory.
    """
    initial_states = [98, 99]  # HOMO
    final_states = tee(range(100, 104), 2)
    
    plams.init()
    project_name = 'spectrum_pentacene'

    cell = [[16.11886919, 0.07814137, -0.697284243],
            [-0.215317662, 4.389405268, 1.408951791],
            [-0.216126961, 1.732808365, 9.748961085]]
    # create Settings for the Cp2K Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = cell
    cp2k_args.specific.cp2k.force_eval.dft.scf.added_mos = 100
    cp2k_args.specific.cp2k.force_eval.dft.scf.diagonalization.jacobi_threshold = 1e-6

    # Setting to calculate the WF use as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_OT.potential = "GTH-PBE"
    cp2k_OT.cell_parameters = cell
    cp2k_OT.specific.cp2k.force_eval.dft.scf.scf_guess = 'atomic'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.minimizer = 'DIIS'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.n_diis = 7
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.added_mos = 0
    cp2k_OT.specific.cp2k.force_eval.dft.scf.eps_scf = 5e-06

    # Path to the MD geometries
    path_traj_xyz = "./pentanceOpt.xyz"

    # User variables
    home = os.path.expanduser('~')  # HOME Path
    username = getpass.getuser()
    
    # Work_dir
    scratch = "/scratch-shared"
    scratch_path = join(scratch, username, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # Cp2k configuration files
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    cp2k_config = {"basis": basiscp2k, "potential": potcp2k}

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # all_geometries type :: [String]
    geometry = split_file_geometries(path_traj_xyz)

    # Hamiltonian computation
    simulate_absoprtion_spectrum('cp2k', project_name, geometry, cp2k_args,
                                 guess_args=cp2k_OT,
                                 initial_states=initial_states,
                                 final_states=final_states,
                                 calc_new_wf_guess_on_points=[0],
                                 path_hdf5=path_hdf5,
                                 package_config=cp2k_config)


# ===================================<>========================================
if __name__ == "__main__":
    main()
