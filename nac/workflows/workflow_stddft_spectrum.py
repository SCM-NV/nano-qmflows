__all__ = ['workflow_stddft']

from nac.common import change_mol_units
from nac.workflows.initialization import initialize
from nac.schedule.components import calculate_mos
from qmflows.parsers import parse_string_xyz
from qmflows import run

from noodles import (gather, schedule)
import h5py
import numpy as np
from typing import (Dict, List)


def workflow_stddft(workflow_settings: Dict):
    """
    Compute the excited states using simplified TDDFT

    :param workflow_settings: Arguments to compute the oscillators see:
    `data/schemas/absorption_spectrum.json
    :returns: None
    """
    # Arguments to compute the orbitals and configure the workflow. see:
    # `data/schemas/general_settings.json
    config = workflow_settings['general_settings']

    # Dictionary containing the general configuration
    config.update(initialize(**config))

    # Single Point calculations settings using CP2K
    mo_paths_hdf5 = calculate_mos(**config)

    # Read structures
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in config['geometries']]

    # Noodles promised call
    scheduleTDDFT = schedule(compute_excited_states_tddft)

    results = gather(
       *[scheduleTDDFT(
           i, mol, config['project_name'], config['package_name'], config['basis_name'],
           mo_paths_hdf5, config['path_hdf5'], workflow_settings['xc_dft'],
           workflow_settings['ci_range'], workflow_settings['nHOMO'],
           workflow_settings['tddft'], config['runner'])
         for i, mol in enumerate(molecules_au)
         if i % workflow_settings['calculate_oscillator_every'] == 0])

    run(results, folder=config['work_dir'])


def compute_excited_states_tddft(
           i: int, mol: List, project_name: str, package_name: str,
           basis_name: str, mo_paths_hdf5, path_hdf5: str, xc_dft: str, ci_range: list,
           nocc: int, tddft: str, runner: str):

    from nac.common import retrieve_hdf5_data

    e, c_ao = retrieve_hdf5_data(path_hdf5, mo_paths_hdf5[i])

    # Number of virtual orbitals
    nvirt = c_ao.shape[1] - nocc

    # Call the function that computes overlaps
    s = getMultipoleMtx(
        mol, package_name, basis_name, path_hdf5, runner, 'overlap')

    # Make a function tha returns in transition density charges
    q = transition_density_charges(mol, s, c_ao)

    # Make a function that compute the Mataga-Nishimoto-Ohno_Klopman damped Columb and Excgange law functions
#    gamma_J, gamma_K = compute_MNOK_integrals(mol, ax, alpha1, alpha2, beta1, beta2)
    gamma_J, gamma_K = compute_MNOK_integrals(mol, xc_dft)

    # Compute the Couloumb and Exchange integrals
    pqrs_J = np.tensordot(q, np.tensordot(q, gamma_J, axes=(0, 1)), axes=(0, 2))
    pqrs_K = np.tensordot(q, np.tensordot(q, gamma_K, axes=(0, 1)), axes=(0, 2))

    # Construct the Tamm-Dancoff matrix A for each pair of i->a transition
    a_mat = construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt)

    if tddft == 'stddft':
        print('sTDDFT has not been implemented yet !')
    # Solve the eigenvalue problem = A * cis = omega * cis
    elif tddft == 'stda':
        omega, xia = np.linalg.eig(a_mat)
    else:
        msg = "Only the stda method is available"
        raise RuntimeError(msg)

    # Compute oscillator strengths

    # 1) Get the pre-factor for each electronic excited state i->a. Size: n_exc_states * Nocc * Nvirt
    e_diff = -np.subtract(e[:nocc].reshape(nocc, 1), e[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)
    pre_factor = np.sqrt(
        2 * np.divide(e_diff.reshape(nocc*nvirt, 1),
                      omega.reshape(1, nocc*nvirt))).T.reshape(nocc*nvirt, nocc, nvirt)

    # 2) Compute the transition dipole matrix TDM(i->a) 
    # Call the function that computes transition dipole moments integrals
    tdm = getMultipoleMtx(
        mol, package_name, basis_name, path_hdf5, runner, 'dipole')
    tdmatrix_x = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[0, :, :], c_ao[:, nocc:]])
    tdmatrix_y = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[1, :, :], c_ao[:, nocc:]])
    tdmatrix_z = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[2, :, :], c_ao[:, nocc:]])

    # 3) Compute the transition dipole moments for each excited state i->a. Size: n_exc_states
    xia_new = xia.reshape(nocc, nvirt, nocc*nvirt)
    d_x = np.hstack(np.trace(np.linalg.multi_dot([pre_factor[i, :, :], xia_new[:, :, i].T, tdmatrix_x])) for i in range(nocc*nvirt))
    d_y = np.hstack(np.trace(np.linalg.multi_dot([pre_factor[i, :, :], xia_new[:, :, i].T, tdmatrix_y])) for i in range(nocc*nvirt))
    d_z = np.hstack(np.trace(np.linalg.multi_dot([pre_factor[i, :, :], xia_new[:, :, i].T, tdmatrix_z])) for i in range(nocc*nvirt))

    # 4) Compute the oscillator strength
    f = 2 / 3 * omega * (d_x ** 2 + d_y ** 2 + d_z ** 2)

    # Write to output
    output = write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, e)
    np.savetxt('output_{}.txt'.format(i), output, fmt='%5d %10.3f %10.5f %10.5f %10.5f %10.5f %10.5f %3d %10.3f %3d %10.3f %10.3f')


# Retrieve some useful information from data
def write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, e):

    from nac.common import h2ev

    excs = []
    for i in range(nocc):
        for a in range(nocc, nvirt+nocc):
            excs.append((i, a))

    output = np.empty((nocc*nvirt, 12))
    output[:, 0] = 0  # State number: we update it after reorder
    output[:, 1] = omega * h2ev  # State energy in eV
    output[:, 2] = f  # Oscillator strength
    output[:, 3] = d_x  # Transition dipole moment in the x direction
    output[:, 4] = d_y  # Transition dipole moment in the y direction
    output[:, 5] = d_z  # Transition dipole moment in the z direction
    # Weight of the most important excitation
    output[:, 6] = np.hstack(np.max(xia[:, i] ** 2) for i in range(nocc*nvirt))

    # Find the index of this transition
    index_weight = np.hstack(
        np.where(xia[:, i] ** 2 == np.max(xia[:, i] ** 2))
        for i in range(nocc*nvirt)).reshape(nocc*nvirt)

    # Index of the hole for the most important excitation
    output[:, 7] = np.stack(excs[index_weight[i]][0] for i in range(nocc*nvirt))
    # These are the energies of the hole for the transition with the larger weight
    output[:, 8] = e[output[:, 7].astype(int)] * h2ev
    # Index of the electron for the most important excitation
    output[:, 9] = np.stack(excs[index_weight[i]][1] for i in range(nocc*nvirt))
    # These are the energies of the electron for the transition with the larger weight
    output[:, 10] = e[output[:, 9].astype(int)] * h2ev
    # This is the energy for the transition with the larger weight
    output[:, 11] = ([output[:, 9].astype(int)] - e[output[:, 7].astype(int)]) * h2ev

    # Reorder the output in ascending order of energy
    output = output[output[:, 1].argsort()]
    # Give a state number in the correct order
    output[:, 0] = np.arange(nocc * nvirt)

    return output


def getMultipoleMtx(mol, package_name, basis_name, path_hdf5, runner, multipole):

    from nac.basisSet import create_dict_CGFs
    from nac.common import (triang2mtx, search_data_in_hdf5, store_arrays_in_hdf5)
    from nac.integrals import (calcMtxOverlapP, calc_transf_matrix)
    from nac.integrals.multipoleIntegrals import calcMtxMultipoleP
    from nac.basisSet.basisNormalization import compute_normalization_sphericals
    from qmflows.parsers.xyzParser import readXYZ
    from scipy import sparse
    from os.path import join

    root = join(project_name, 'multipole')

    # Compute the number of cartesian basis functions
    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, mol)
    n_cart_funcs = np.sum(np.stack(len(dictCGFs[mol[i].symbol]) for i in range(len(mol))))

    # Compute the transformation matrix from cartesian to spherical
    dict_global_norms = compute_normalization_sphericals(dictCGFs)
    with h5py.File(path_hdf5, 'r') as f5:
        transf_mtx = calc_transf_matrix(
             f5, mol, basis_name, dict_global_norms, package_name)
    transf_mtx = sparse.csr_matrix(transf_mtx)
    transpose = transf_mtx.transpose()

    if multipole == 'overlap':
        overlaps_paths_hdf5 = join(root, 'overlaps')
        if search_data_in_hdf5(path_hdf5, overlaps_paths_hdf5):
            with h5py.File(path_hdf5, 'r') as f5:
                m = f5['{}'.format(overlaps_paths_hdf5)].value
                print('Retrieving overlap from hdf5')
        else:
            print('Computing overlap')
            rs = calcMtxOverlapP(mol, dictCGFs)
            mtx_overlap = triang2mtx(rs, n_cart_funcs)  # there are 1452 Cartesian basis CGFs
            m = transf_mtx.dot(sparse.csr_matrix.dot(mtx_overlap, transpose))
            store_arrays_in_hdf5(path_hdf5, overlaps_paths_hdf5, m)

    elif multipole == 'dipole':
        dipole_paths_hdf5 = join(root, 'dipole')
        if search_data_in_hdf5(path_hdf5, dipole_paths_hdf5):
            with h5py.File(path_hdf5, 'r') as f5:
                m = f5['{}'.format(dipole_paths_hdf5)].value
                print('Retrieving transition dipole matrix from hdf5')
        else:
            print('Computing transition dipole matrix')
            rc = (0, 0, 0)
            exponents = [{'e': 1, 'f': 0, 'g': 0},
                         {'e': 0, 'f': 1, 'g': 0},
                         {'e': 0, 'f': 0, 'g': 1}]
            mtx_integrals_triang = tuple(calcMtxMultipoleP(mol, dictCGFs, runner, rc, **kw)
                                         for kw in exponents)
            mtx_integrals_cart = tuple(triang2mtx(xs, n_cart_funcs)
                                       for xs in mtx_integrals_triang)
            m = np.stack(transf_mtx.dot(sparse.csr_matrix.dot(x, transpose)) for x in mtx_integrals_cart)
            store_arrays_in_hdf5(path_hdf5, dipole_paths_hdf5, m)

    elif multipole == 'quadrupole':
        quadrupole_paths_hdf5 = join(root, 'quadrupole')
        if search_data_in_hdf5(path_hdf5, quadrupole_paths_hdf5):
            with h5py.File(path_hdf5, 'r') as f5:
                m = f5['{}'.format(quadrupole_paths_hdf5)].value
                print('Retrieving transition quadrupole matrix from hdf5')
        else:
            print('Computing transition quadrupole matrix')
            rc = (0, 0, 0)
            exponents = [{'e': 2, 'f': 0, 'g': 0},
                         {'e': 0, 'f': 2, 'g': 0},
                         {'e': 0, 'f': 0, 'g': 2}]
            mtx_integrals_triang = tuple(calcMtxMultipoleP(mol, dictCGFs, runner, rc, **kw)
                                         for kw in exponents)
            mtx_integrals_cart = tuple(triang2mtx(xs, n_cart_funcs)
                                       for xs in mtx_integrals_triang)
            m = np.stack(transf_mtx.dot(sparse.csr_matrix.dot(x, transpose)) for x in mtx_integrals_cart)
            store_arrays_in_hdf5(path_hdf5, quadrupole_paths_hdf5, m)

            return m

def n_sph_funcs_per_atom(mol, package_name, basis_name, path_hdf5):

    from nac.integrals.spherical_Cartesian_cgf import (calc_orbital_Slabels, read_basis_format)

    with h5py.File(path_hdf5, 'r') as f5:
        xs = [f5['{}/basis/{}/{}/coefficients'.format(package_name, mol[i][0], basis_name)] for i in range(len(mol))]
        ys = [calc_orbital_Slabels(package_name, read_basis_format(package_name, xs[i].attrs['basisFormat'])) for i in range(len(mol))]

    return np.stack(np.sum(len(x) for x in ys[i]) for i in range(len(mol)))


def transition_density_charges(mol, s, c_ao):
    from scipy.linalg import sqrtm
   
    n_atoms = len(mol)
    sqrt_s = sqrtm(s)
    c_mo = np.dot(sqrt_s, c_ao)
    q = np.zeros((n_atoms, c_mo.shape[1], c_mo.shape[1])) # Size of the transition density tensor : n_atoms x n_mos x n_mos
    n_sph_atoms = n_sph_funcs_per_atom(mol, package_name, basis_name, path_hdf5)
    
    index = 0
    for i in range(n_atoms):
        q[i, :, :] = np.dot(c_mo[index:(index + n_sph_atoms[i]), :].T, c_mo[index:(index + n_sph_atoms[i]), :])
        index += n_sph_atoms[i]

    return q 


def compute_MNOK_integrals(mol, xc_dft):
    from scipy.spatial.distance import cdist
    from nac.common import (hardness, xc)

    n_atoms = len(mol)
    coords = np.asarray([mol[i][1] for i in range(len(mol))])
    r_ab = cdist(coords, coords) # Distance matrix between atoms A and B
    hardness_vec = np.stack(hardness(mol[i][0]) for i in range(n_atoms)).reshape(n_atoms, 1)
    hard = np.dot(hardness_vec, hardness_vec.T) / 2
    beta = xc[xc_dft]['beta1'] + xc[xc_dft]['ax'] * xc[xc_dft]['beta2']
    alpha = xc[xc_dft]['alpha1'] + xc[xc_dft]['ax'] * xc[xc_dft]['alpha2']
    gamma_J = np.power(1 / (np.power(r_ab, beta) + ax * np.power(hard, -beta)), 1/beta)
    gamma_J[gamma_J == np.inf] = 0 # When ax = 0 , you can get infinite values on the diagonal. Just turn them off to 0.
    gamma_K = np.power(1 / (np.power(r_ab, alpha) + np.power(hard, -alpha)), 1/alpha)

    return gamma_J, gamma_K


def construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt):

    # This is the exchange integral entering the A matrix.
    #  It is in the format (nocc, nvirt, nocc, nvirt)
    k_iajb = 2 * pqrs_K[:nocc, nocc:, :nocc, nocc:].reshape(nocc*nvirt, nocc*nvirt)

    # This is the Coulomb integral entering in the A matrix.
    # It is in the format: (nocc, nocc, nvirt, nvirt)
    k_ijab_tmp = ax * pqrs_J[:nocc, :nocc, nocc:, nocc:]

    # To get the correct order in the A matrix, i.e. (nocc, nvirt, nocc, nvirt),
    # we have to swap axes
    k_ijab = np.swapaxes(k_ijab_tmp, axis1=1, axis2=2).reshape(nocc*nvirt, nocc*nvirt)

    # They are in the m x m format where m is the number of excitations = nocc * nvirt
    a_mat = k_iajb - k_ijab

    # Generate a vector with all possible ea - ei energy differences
    e_diff = -np.subtract(
        e[:nocc].reshape(nocc, 1), e[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)
    np.fill_diagonal(a_mat, np.diag(a_mat) + e_diff)

    return a_mat
