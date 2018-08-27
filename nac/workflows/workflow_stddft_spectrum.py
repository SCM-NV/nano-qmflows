__all__ = ['workflow_stddft']

from nac.common import (
    change_mol_units, h2ev, hardness, retrieve_hdf5_data, xc)
from nac.integrals.multipole_matrices import get_multipole_matrix
from nac.integrals.spherical_Cartesian_cgf import (calc_orbital_Slabels, read_basis_format)
from nac.schedule.components import calculate_mos
from nac.workflows.initialization import initialize
from qmflows.parsers import parse_string_xyz
from qmflows import run
from noodles import (gather, schedule)
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from typing import (Dict, List)
import h5py
import numpy as np


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
           i, mol, mo_paths_hdf5, workflow_settings['xc_dft'],
           workflow_settings['ci_range'], workflow_settings['nHOMO'],
           workflow_settings['tddft'], config)
         for i, mol in enumerate(molecules_au)
         if i % workflow_settings['calculate_oscillator_every'] == 0])

    run(results, folder=config['work_dir'])


def compute_excited_states_tddft(
           i: int, mol: List, mo_paths_hdf5, xc_dft: str, ci_range: list,
           nocc: int, tddft: str, config: Dict):
    """
    ADD DOCUMENTATION
    """
    e, c_ao = retrieve_hdf5_data(config['path_hdf5'], mo_paths_hdf5[i])

    # Number of virtual orbitals
    nvirt = c_ao.shape[1] - nocc

    # Call the function that computes overlaps
    s = get_multipole_matrix(
        mol, config, 'overlap')

    # Make a function tha returns in transition density charges
    q = transition_density_charges(mol, config, s, c_ao)

    # Make a function that compute the Mataga-Nishimoto-Ohno_Klopman damped Columb and Excgange law functions
    gamma_J, gamma_K = compute_MNOK_integrals(mol, xc_dft)

    # Compute the Couloumb and Exchange integrals
    pqrs_J = np.tensordot(q, np.tensordot(q, gamma_J, axes=(0, 1)), axes=(0, 2))
    pqrs_K = np.tensordot(q, np.tensordot(q, gamma_K, axes=(0, 1)), axes=(0, 2))

    # Construct the Tamm-Dancoff matrix A for each pair of i->a transition
    a_mat = construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt, xc_dft, e)

    if tddft == 'stddft':
        print('sTDDFT has not been implemented yet !')
    # Solve the eigenvalue problem = A * cis = omega * cis
    elif tddft == 'stda':
        omega, xia = np.linalg.eig(a_mat)
    else:
        msg = "Only the stda method is available"
        raise RuntimeError(msg)
    # Compute oscillator strengths
    # The formula can be rearranged like this:
    # f_I = 2/3 * np.sqrt(2 * omega_I) * sum_ia ( np.sqrt(e_diff_ia) * xia * tdm_x) ** 2 + y^2 + z^2

    # 1) Get the energy matrix i->a. Size: Nocc * Nvirt
    delta_ia = -np.subtract(e[:nocc].reshape(nocc, 1), e[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)

    # 2) Compute the transition dipole matrix TDM(i->a)
    # Call the function that computes transition dipole moments integrals
    tdm = get_multipole_matrix(mol, config, 'dipole')
    tdmatrix_x = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[0, :, :], c_ao[:, nocc:]]).reshape(nocc*nvirt)
    tdmatrix_y = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[1, :, :], c_ao[:, nocc:]]).reshape(nocc*nvirt)
    tdmatrix_z = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[2, :, :], c_ao[:, nocc:]]).reshape(nocc*nvirt)

    # 3) Compute the transition dipole moments for each excited state i->a. Size: n_exc_states
    d_x = np.stack(np.sum(np.sqrt(delta_ia) * xia[:, i] * tdmatrix_x) for i in range(nocc*nvirt))
    d_y = np.stack(np.sum(np.sqrt(delta_ia) * xia[:, i] * tdmatrix_y) for i in range(nocc*nvirt))
    d_z = np.stack(np.sum(np.sqrt(delta_ia) * xia[:, i] * tdmatrix_z) for i in range(nocc*nvirt))

    # 4) Compute the oscillator strength
    f = 2 / 3 * np.sqrt(2 * omega) * (d_x ** 2 + d_y ** 2 + d_z ** 2)

    # Write to output
    output = write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, e)
    np.savetxt('output_{}.txt'.format(i), output, fmt='%5d %10.3f %10.5f %10.5f %10.5f %10.5f %10.5f %3d %10.3f %3d %10.3f %10.3f')


def write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, e):
    """ Write out as a table in plane text"""

    excs = []
    for i in range(nocc):
        for a in range(nocc, nvirt + nocc):
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
        np.where(xia[:, i] ** 2 == np.max(xia[:, i] ** 2)) for i in range(nocc*nvirt)).reshape(nocc*nvirt)

    # Index of the hole for the most important excitation
    output[:, 7] = np.stack(excs[index_weight[i]][0] for i in range(nocc*nvirt)) + 1
    # These are the energies of the hole for the transition with the larger weight
    output[:, 8] = e[output[:, 7].astype(int) - 1] * h2ev
    # Index of the electron for the most important excitation
    output[:, 9] = np.stack(excs[index_weight[i]][1] for i in range(nocc*nvirt)) + 1
    # These are the energies of the electron for the transition with the larger weight
    output[:, 10] = e[output[:, 9].astype(int) - 1] * h2ev
    # This is the energy for the transition with the larger weight
    output[:, 11] = (e[output[:, 9].astype(int) - 1] - e[output[:, 7].astype(int) - 1]) * h2ev

    # Reorder the output in ascending order of energy
    output = output[output[:, 1].argsort()]
    # Give a state number in the correct order
    output[:, 0] = np.arange(nocc * nvirt) + 1

    return output


def number_spherical_functions_per_atom(mol, package_name, basis_name, path_hdf5):
    """
    ADD Documentation
    """
    with h5py.File(path_hdf5, 'r') as f5:
        xs = [f5['{}/basis/{}/{}/coefficients'.format(package_name, atom[0], basis_name)] for atom in mol]
        ys = [calc_orbital_Slabels(
            package_name, read_basis_format(package_name, path.attrs['basisFormat'])) for path in xs]

    return np.stack(np.sum(len(x) for x in ys[i]) for i in range(len(mol)))


def transition_density_charges(mol, config, s, c_ao):
    """
    ADD Documentation
    """
    n_atoms = len(mol)
    sqrt_s = sqrtm(s)
    c_mo = np.dot(sqrt_s, c_ao)
    # Size of the transition density tensor : n_atoms x n_mos x n_mos
    q = np.zeros((n_atoms, c_mo.shape[1], c_mo.shape[1]))
    n_sph_atoms = number_spherical_functions_per_atom(
        mol, config['package_name'], config['basis_name'], config['path_hdf5'])

    index = 0
    for i in range(n_atoms):
        q[i, :, :] = np.dot(c_mo[index:(index + n_sph_atoms[i]), :].T, c_mo[index:(index + n_sph_atoms[i]), :])
        index += n_sph_atoms[i]

    return q


def compute_MNOK_integrals(mol, xc_dft):
    """
    ADD Documentation
    """
    n_atoms = len(mol)
    coords = np.asarray([mol[i][1] for i in range(len(mol))])
    # Distance matrix between atoms A and B
    r_ab = cdist(coords, coords)
    hardness_vec = np.stack(hardness(mol[i][0]) for i in range(n_atoms)).reshape(n_atoms, 1)
    hard = np.add(hardness_vec, hardness_vec.T)
    beta = xc(xc_dft)['beta1'] + xc(xc_dft)['ax'] * xc(xc_dft)['beta2']
    alpha = xc(xc_dft)['alpha1'] + xc(xc_dft)['ax'] * xc(xc_dft)['alpha2']
    if (xc(xc_dft)['ax'] == 0):
        gamma_J = 1 / r_ab
        gamma_J[gamma_J == np.inf] = 0
    else:
        gamma_J = np.power(
            1 / (np.power(r_ab, beta) + np.power((xc(xc_dft)['ax'] * hard), -beta)), 1/beta)
    gamma_K = np.power(1 / (np.power(r_ab, alpha) + np.power(hard, -alpha)), 1/alpha)

    return gamma_J, gamma_K


def construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt, xc_dft, e):
    """
    ADD Documentation
    """
    # This is the exchange integral entering the A matrix.
    #  It is in the format (nocc, nvirt, nocc, nvirt)
    k_iajb = pqrs_K[:nocc, nocc:, :nocc, nocc:].reshape(nocc*nvirt, nocc*nvirt)

    # This is the Coulomb integral entering in the A matrix.
    # It is in the format: (nocc, nocc, nvirt, nvirt)
    k_ijab_tmp = pqrs_J[:nocc, :nocc, nocc:, nocc:]

    # To get the correct order in the A matrix, i.e. (nocc, nvirt, nocc, nvirt),
    # we have to swap axes
    k_ijab = np.swapaxes(k_ijab_tmp, axis1=1, axis2=2).reshape(nocc*nvirt, nocc*nvirt)

    # They are in the m x m format where m is the number of excitations = nocc * nvirt
    a_mat = 2 * k_iajb - k_ijab

    # Generate a vector with all possible ea - ei energy differences
    e_diff = -np.subtract(
        e[:nocc].reshape(nocc, 1), e[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)
    np.fill_diagonal(a_mat, np.diag(a_mat) + e_diff)

    return a_mat
