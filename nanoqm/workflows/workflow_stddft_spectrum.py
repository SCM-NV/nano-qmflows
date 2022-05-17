"""Compute the excited states energies and MO coefficients using the STDDFT approach.

Index
-----
.. currentmodule:: nanoqm.workflows.workflow_stddft_spectrum
.. autosummary::
    workflow_stddft

"""

from __future__ import annotations

__all__ = ['workflow_stddft']

import warnings
from os.path import join
from typing import Tuple, TYPE_CHECKING

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from noodles import gather, schedule, unpack
from noodles.interface import PromisedObject
from qmflows.parsers import parse_string_xyz
from qmflows.warnings_qmflows import Orbital_Warning
from qmflows.type_hints import PathLike

from .. import logger
from ..common import (DictConfig, angs2au, change_mol_units, h2ev, hardness,
                      is_data_in_hdf5, number_spherical_functions_per_atom,
                      retrieve_hdf5_data, store_arrays_in_hdf5, xc)
from ..integrals.multipole_matrices import get_multipole_matrix
from ..schedule.components import calculate_mos
from .orbitals_type import select_orbitals_type

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float64 as f8


def workflow_stddft(config: DictConfig) -> None:
    """Compute the excited states using simplified TDDFT.

    Both restricted and unrestricted orbitals calculations are available.

    Parameters
    ----------
    config
        Dictionary with the configuration to run the workflows

    """
    return select_orbitals_type(config, run_workflow_stddft)


def run_workflow_stddft(config: DictConfig) -> PromisedObject:
    """Compute the excited states using simplified TDDFT using `config`."""
    # Single Point calculations settings using CP2K
    mo_paths_hdf5, energy_paths_hdf5 = unpack(calculate_mos(config), 2)

    # Read structures
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for i, gs in enumerate(config.geometries)
                    if (i % config.stride) == 0]

    # Noodles promised call
    scheduleTDDFT = schedule(compute_excited_states_tddft)

    results = gather(
        *[scheduleTDDFT(config, mo_paths_hdf5[i], DictConfig(
            {'i': i * config.stride, 'mol': mol}))
          for i, mol in enumerate(molecules_au)])

    return gather(results, energy_paths_hdf5)


def validate_active_space(
    config: DictConfig,
    nocc_molog: int,
    nvirt_molog: int,
) -> tuple[int, int]:
    """Validate and return the number of occupied and virtual MOs.

    The ``active_space`` keyword is treated as an upper bound by CP2K, so its values
    might be larger than the actual available number of occupied and/or virtual MOs.

    Parameters
    ----------
    config : DictConfig
        The Nano-QMFlows configuration.
    nocc_molog/nvirt_molog : int
        The number of occupied and virtual orbitals as extracted from the MOLog orbital file.

    Returns
    -------
    tuple[int, int]
        The (corrected) number of occupied and virtual orbitals.

    """
    nocc_inp = config.active_space[0]
    nvirt_inp = config.active_space[1]

    # Correct for the presence of SOMOs, which translates to either additional
    # occupied or unoccupied orbitals depending on the spin
    if config.orbitals_type == "alphas":
        nocc_inp += (config.multiplicity - 1)
    elif config.orbitals_type == "beta":
        nvirt_inp += (config.multiplicity - 1)

    if (nocc_inp == nocc_molog) and (nvirt_inp == nvirt_molog):
        return nocc_inp, nvirt_inp
    else:
        config.active_space = [nocc_molog, nvirt_molog]

    # The input and MOlog file have different numbers of active and/or virtual orbitals;
    # issue a warning and return the correct number
    mo_type_list = []
    if nocc_inp > nocc_molog:
        mo_type_list.append("occupied")
    if nvirt_inp > nvirt_molog:
        mo_type_list.append("virtual")
    mo_type = config.orbitals_type + " " if config.orbitals_type in ("alphas", "betas") else ""
    mo_type += "and ".join(f"{i} " for i in mo_type_list)

    warnings.warn(
        f"The requested activate space is larger than the available number of {mo_type}MOs; "
        "adjusting active space", Orbital_Warning,
    )
    return nocc_molog, nvirt_molog


def compute_excited_states_tddft(
        config: DictConfig, path_MOs: list[str], dict_input: DictConfig) -> None:
    """Compute the excited states properties (energy and coefficients).

    Take a given `mo_index_range`, the `tddft` method and `xc_dft` exchange functional.
    """
    logger.info("Reading energies and mo coefficients")

    # type of calculation
    energy, c_ao, nocc_nvirt = retrieve_hdf5_data(config.path_hdf5, path_MOs)

    # Number of virtual orbitals
    nocc, nvirt = validate_active_space(config, *nocc_nvirt)
    dict_input.update({"energy": energy, "c_ao": c_ao,
                       "nocc": nocc, "nvirt": nvirt})

    # Pass the molecule in Angstrom to the libint calculator
    copy_dict = DictConfig(dict_input.copy())
    copy_dict["mol"] = change_mol_units(dict_input["mol"], factor=1 / angs2au)

    # compute the multipoles if they are not stored
    multipoles = get_multipole_matrix(config, copy_dict, 'dipole')

    # read data from the HDF5 or calculate it on the fly
    dict_input["overlap"] = multipoles[0]

    # retrieve or compute the omega xia values
    omega, xia = get_omega_xia(config, dict_input)

    # add arrays to the dictionary
    dict_input.update(
        {"multipoles": multipoles[1:], "omega": omega, "xia": xia})

    compute_oscillator_strengths(config, dict_input)


def get_omega_xia(
        config: DictConfig, dict_input: DictConfig) -> Tuple[NDArray[f8], NDArray[f8]]:
    """Search for the multipole_matrices, Omega and xia values in the HDF5.

    if they are not available compute and store them.

    Returns
    -------
    tuple
        omega and xia numpy arrays.

    """
    tddft = config.tddft.lower()

    def compute_omega_xia():
        if tddft == 'sing_orb':
            return compute_sing_orb(dict_input)

        return compute_std_aproximation(config, dict_input)

    # search data in HDF5
    point = f'point_{dict_input.i + config.enumerate_from}'

    paths_omega_xia = [join(x, point) for x in ("omega", "xia")]

    if is_data_in_hdf5(config.path_hdf5, paths_omega_xia):
        ret = retrieve_hdf5_data(config.path_hdf5, paths_omega_xia)
        return ret[0], ret[1]

    else:
        omega, xia = compute_omega_xia()
        store_arrays_in_hdf5(
            config.path_hdf5, paths_omega_xia[0], omega, dtype=omega.dtype)
        store_arrays_in_hdf5(
            config.path_hdf5, paths_omega_xia[1], xia, dtype=xia.dtype)

        return omega, xia


def compute_sing_orb(inp: DictConfig) -> Tuple[NDArray[f8], NDArray[f8]]:
    """Compute the Single Orbital approximation."""
    energy, nocc, nvirt = tuple(inp[x]for x in ("energy", "nocc", "nvirt"))

    omega = energy[:nocc][..., None] - energy[nocc:][..., None].T
    xia = np.eye(omega.size)
    return -omega.ravel(), xia


def compute_std_aproximation(
        config: DictConfig, dict_input: DictConfig) -> Tuple[NDArray[f8], NDArray[f8]]:
    """Compute the oscillator strenght using either the stda or stddft approximations."""
    logger.info("Reading or computing the dipole matrices")

    # Make a function tha returns in transition density charges
    logger.info("Computing the transition density charges")
    # multipoles[0] is the overlap matrix
    q = transition_density_charges(
        dict_input.mol, config, dict_input.overlap, dict_input.c_ao)

    # Make a function that compute the Mataga-Nishimoto-Ohno_Klopman
    # damped Columb and Excgange law functions
    logger.info(
        "Computing the gamma functions for Exchange and Coulomb integrals")
    gamma_J, gamma_K = compute_MNOK_integrals(dict_input["mol"], config.xc_dft)

    # Compute the Couloumb and Exchange integrals
    # If xc_dft is a pure functional, ax=0, thus the pqrs_J ints are not needed
    # and can be set to 0
    logger.info("Computing the Exchange and Coulomb integrals")
    if (xc(config.xc_dft)['type'] == 'pure'):
        size = dict_input.energy.size
        pqrs_J = np.zeros((size, size, size, size))
    else:
        pqrs_J = np.tensordot(q, np.tensordot(
            q, gamma_J, axes=(0, 1)), axes=(0, 2))
    pqrs_K = np.tensordot(q, np.tensordot(
        q, gamma_K, axes=(0, 1)), axes=(0, 2))

    # Construct the Tamm-Dancoff matrix A for each pair of i->a transition
    logger.info("Constructing the A matrix for TDDFT calculation")
    a_mat = construct_A_matrix_tddft(
        pqrs_J, pqrs_K, dict_input.nocc, dict_input.nvirt, config.xc_dft, dict_input.energy)

    if config.tddft == 'stddft':
        logger.info('sTDDFT has not been implemented yet !')
        # Solve the eigenvalue problem = A * cis = omega * cis
    elif config.tddft == 'stda':
        logger.info(
            "This is a TDA calculation ! \n Solving the eigenvalue problem")
        omega, xia = np.linalg.eig(a_mat)
    else:
        msg = f"The {config.tddft} method has not been implemented"
        raise NotImplementedError(msg)

    return omega, xia


def compute_oscillator_strengths(config: DictConfig, inp: DictConfig) -> None:
    """Compute oscillator strengths.

    The formula can be rearranged like this:
    f_I = 2/3 * np.sqrt(2 * omega_I) * sum_ia ( np.sqrt(e_diff_ia) * xia * tdm_x) ** 2 + y^2 + z^2
    """
    tddft = config.tddft.lower()
    # 1) Get the inp.energy matrix i->a. Size: Inp.Nocc * Inp.Nvirt
    delta_ia = -np.subtract(
        inp.energy[:inp.nocc].reshape(inp.nocc, 1),
        inp.energy[inp.nocc:].reshape(inp.nvirt, 1).T).reshape(inp.nocc * inp.nvirt)

    def compute_transition_matrix(matrix):
        if tddft == 'sing_orb':
            tm = np.stack(
                [np.sum(
                 delta_ia / inp.omega[i] * inp.xia[:, i] * matrix)
                 for i in range(inp.nocc * inp.nvirt)])
        else:
            tm = np.stack(
                [np.sum(
                 np.sqrt(2 * delta_ia / inp.omega[i]) * inp.xia[:, i] * matrix)
                 for i in range(inp.nocc * inp.nvirt)])
        return tm

    # 2) Compute the transition dipole matrix TDM(i->a)
    # Call the function that computes transition dipole moments integrals
    logger.info("Reading or computing the transition dipole matrix")

    def compute_tdmatrix(k):
        return np.linalg.multi_dot(
            [inp.c_ao[:, :inp.nocc].T, inp.multipoles[k, :, :],
             inp.c_ao[:, inp.nocc:]]).reshape(inp.nocc * inp.nvirt)

    td_matrices = (compute_tdmatrix(k) for k in range(3))

    # 3) Compute the transition dipole moments for each excited state i->a. Size: n_exc_states
    d_x, d_y, d_z = tuple(
        compute_transition_matrix(m) for m in td_matrices)

    # 4) Compute the oscillator strength
    f = 2 / 3 * inp.omega * (d_x ** 2 + d_y ** 2 + d_z ** 2)

    # Write to output
    inp.update({"dipole": (d_x, d_y, d_z), "oscillator": f})
    write_output(config, inp)


def write_output(config: DictConfig, inp: DictConfig) -> None:
    """Write the results using numpy functionality."""
    output = write_output_tddft(inp)

    path_output = join(config.workdir,
                       f'output_{inp.i + config.enumerate_from}_{config.tddft}.txt')
    fmt = '{:^5s}{:^14s}{:^8s}{:^11s}{:^11s}{:^11s}{:^11s}{:<5s}{:^10s}{:<5s}{:^11s}{:^11s}'
    header = fmt.format(
        'state', 'energy', 'f', 't_dip_x', 't_dip_y', 't_dip_y', 'weight',
        'from', 'energy', 'to', 'energy', 'delta_E')
    np.savetxt(path_output, output,
               fmt='%5d %10.3f %10.5f %10.5f %10.5f %10.5f %10.5f %3d %10.3f %3d %10.3f %10.3f',
               header=header)


def ex_descriptor(omega, f, xia, n_lowest, c_ao, s, tdm, tqm, nocc, nvirt, mol, config):
    """TODO: ADD DOCUMENTATION."""
    # Reshape xia
    xia_I = xia.reshape(nocc, nvirt, nocc * nvirt)

    # Transform the transition density matrix into AO basis
    d0I_ao = np.stack(
        np.linalg.multi_dot(
            [c_ao[:, :nocc], xia_I[:, :, i], c_ao[:, nocc:].T]) for i in range(n_lowest))

    # Compute omega in excition analysis for the lowest n excitations
    om = get_omega(d0I_ao, s, n_lowest)

    # Compute the distribution of positions for the hole and electron
    xh, yh, zh = get_exciton_positions(d0I_ao, s, tdm, n_lowest, 'hole')
    xe, ye, ze = get_exciton_positions(d0I_ao, s, tdm, n_lowest, 'electron')

    # Compute the distribution of the square of position for the hole and electron
    x2h, y2h, z2h = get_exciton_positions(d0I_ao, s, tqm, n_lowest, 'hole')
    x2e, y2e, z2e = get_exciton_positions(d0I_ao, s, tqm, n_lowest, 'electron')

    # Compute the distribution of both hole and electron positions
    xhxe, yhye, zhze = get_exciton_positions(d0I_ao, s, tdm, n_lowest, 'both')

    # Compute Descriptors

    # Compute exciton size:
    d_exc = np.sqrt(
        ((x2h - 2 * xhxe + x2e) + (y2h - 2 * yhye + y2e) + (z2h - 2 * zhze + z2e)) / om)

    # Compute centroid electron_hole distance
    d_he = np.abs(((xe - xh) + (ye - yh) + (ze - zh)) / om)

    # Compute hole and electron size
    sigma_h = np.sqrt(
        (x2h / om - (xh / om) ** 2) + (y2h / om - (yh / om) ** 2) + (z2h / om - (zh / om) ** 2))
    sigma_e = np.sqrt(
        (x2e / om - (xe / om) ** 2) + (y2e / om - (ye / om) ** 2) + (z2e / om - (ze / om) ** 2))

    # Compute Pearson coefficients
    cov = (xhxe - xh * xe) + (yhye - yh * ye) + (zhze - zh * ze)
    r_eh = cov / (sigma_h * sigma_e)

    # Compute approximate d_exc and binding energy
    omega_ab = get_omega_ab(d0I_ao, s, n_lowest, mol, config)
    r_ab = get_r_ab(mol)

    d_exc_apprx = np.stack(
        np.sqrt(np.sum(omega_ab[i, :, :] * (r_ab ** 2)) / om[i]) for i in range(n_lowest))
    # binding energy approximated
    xs = np.stack((omega_ab[i, :, :] / r_ab) for i in range(n_lowest))
    xs[np.isinf(xs)] = 0
    binding_en_apprx = np.stack(
        (np.sum(xs[i, :, :]) / om[i]) for i in range(n_lowest))

    descriptors = write_output_descriptors(
        d_exc, d_exc_apprx, d_he, sigma_h, sigma_e, r_eh, binding_en_apprx, n_lowest, omega, f)

    return descriptors


def write_output_descriptors(
        d_exc, d_exc_apprx, d_he, sigma_h, sigma_e, r_eh, binding_ex_apprx, n_lowest, omega, f):
    """TODO: add Documentation."""
    au2ang = 0.529177249
    ex_output = np.empty((n_lowest, 10))
    ex_output[:, 0] = np.arange(n_lowest) + 1
    ex_output[:, 1] = d_exc * au2ang
    ex_output[:, 2] = d_exc_apprx * au2ang
    ex_output[:, 3] = d_he * au2ang
    ex_output[:, 4] = sigma_h * au2ang
    ex_output[:, 5] = sigma_e * au2ang
    ex_output[:, 6] = r_eh * au2ang
    ex_output[:, 7] = binding_ex_apprx * 27.211 / 2  # in eV
    ex_output[:, 8] = omega[:n_lowest] * 27.211
    ex_output[:, 9] = f[:n_lowest]

    return ex_output


def get_omega(d0I_ao: NDArray[f8], s, n_lowest: int) -> NDArray[f8]:
    """TODO: add Documentation."""
    return np.stack([
        np.trace(np.linalg.multi_dot([d0I_ao[i, :, :].T, s, d0I_ao[i, :, :], s]))
        for i in range(n_lowest)
    ])


def get_r_ab(mol):
    """TODO: add Documentation."""
    coords = np.asarray([atom[1] for atom in mol])
    # Distance matrix between atoms A and B
    r_ab = cdist(coords, coords)
    return r_ab


def get_omega_ab(d0I_ao, s, n_lowest, mol, config):
    """TODO: add Documentation."""
    # Lowdin transformation of the transition density matrix
    n_atoms = len(mol)
    s_sqrt = sqrtm(s)
    d0I_mo = np.stack(
        np.linalg.multi_dot([s_sqrt, d0I_ao[i, :, :], s_sqrt]) for i in range(n_lowest))

    # Compute the number of spherical functions for each atom
    n_sph_atoms = number_spherical_functions_per_atom(
        mol, config['package_name'], config['basis_name'], config['path_hdf5'])

    # Compute omega_ab
    omega_ab = np.zeros((n_lowest, n_atoms, n_atoms))
    for i in range(n_lowest):
        index_a = 0
        for a in range(n_atoms):
            index_b = 0
            for b in range(n_atoms):
                omega_ab[i, a, b] = np.sum(
                    d0I_mo[i, index_a:(index_a + n_sph_atoms[a]),
                           index_b:(index_b + n_sph_atoms[b])] ** 2)
                index_b += n_sph_atoms[b]
            index_a += n_sph_atoms[a]

    return omega_ab


def get_exciton_positions(d0I_ao, s, moment, n_lowest, carrier):
    """TODO: add Documentation."""
    def compute_component_hole(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot([d0I_ao[i, :, :].T, moment[k, :, :], d0I_ao[i, :, :], s]))
            for i in range(n_lowest))

    def compute_component_electron(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot([d0I_ao[i, :, :].T, s, d0I_ao[i, :, :], moment[k, :, :]]))
            for i in range(n_lowest))

    def compute_component_he(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot(
                    [d0I_ao[i, :, :].T, moment[0, :, :], d0I_ao[i, :, :], moment[0, :, :]]))
            for i in range(n_lowest))

    if carrier == 'hole':
        return tuple(compute_component_hole(k) for k in range(3))
    elif carrier == 'electron':
        return tuple(compute_component_electron(k) for k in range(3))
    elif carrier == 'both':
        return tuple(compute_component_he(k) for k in range(3))
    else:
        raise RuntimeError(f"unkown option: {carrier}")


def write_output_tddft(inp: DictConfig) -> np.ndarray:
    """Write out as a table in plane text."""
    energy = inp.energy

    excs = [(i, a) for i in range(inp.nocc)
            for a in range(inp.nocc, inp.nvirt + inp.nocc)]

    output = np.empty((inp.nocc * inp.nvirt, 12))
    output[:, 0] = 0  # State number: we update it after reorder
    output[:, 1] = inp.omega * h2ev  # State energy in eV
    output[:, 2] = inp.oscillator  # Oscillator strength

    d_x, d_y, d_z = inp.dipole
    output[:, 3] = d_x  # Transition dipole moment in the x direction
    output[:, 4] = d_y  # Transition dipole moment in the y direction
    output[:, 5] = d_z  # Transition dipole moment in the z direction
    # Weight of the most important excitation
    output[:, 6] = np.hstack([np.max(inp.xia[:, i] ** 2)
                              for i in range(inp.nocc * inp.nvirt)])

    # Find the index of this transition
    index_weight = np.hstack([
        np.where(
            inp.xia[:, i] ** 2 == np.max(
                inp.xia[:, i] ** 2))
        for i in range(inp.nocc * inp.nvirt)]).reshape(inp.nocc * inp.nvirt)

    # Index of the hole for the most important excitation
    output[:, 7] = np.stack([excs[index_weight[i]][0]
                             for i in range(inp.nocc * inp.nvirt)]) + 1
    # These are the energies of the hole for the transition with the larger weight
    output[:, 8] = energy[output[:, 7].astype(int) - 1] * h2ev
    # Index of the electron for the most important excitation
    output[:, 9] = np.stack([excs[index_weight[i]][1]
                             for i in range(inp.nocc * inp.nvirt)]) + 1
    # These are the energies of the electron for the transition with the larger weight
    output[:, 10] = energy[output[:, 9].astype(int) - 1] * h2ev
    # This is the energy for the transition with the larger weight
    output[:, 11] = (
        energy[output[:, 9].astype(int) - 1] - energy[output[:, 7].astype(int) - 1]) * h2ev

    # Reorder the output in ascending order of energy
    output = output[output[:, 1].argsort()]
    # Give a state number in the correct order
    output[:, 0] = np.arange(inp.nocc * inp.nvirt) + 1

    return output


def transition_density_charges(mol, config, s, c_ao):
    """TODO: add Documentation."""
    n_atoms = len(mol)

    # Due to numerical noise `srtm(s)` can produce a complex array with
    # (approximately) 0-valued imaginary components; get rid of these
    sqrt_s = np.real_if_close(sqrtm(s))
    c_mo = np.dot(sqrt_s, c_ao)

    # Size of the transition density tensor : n_atoms x n_mos x n_mos
    q = np.zeros((n_atoms, c_mo.shape[1], c_mo.shape[1]))
    n_sph_atoms = number_spherical_functions_per_atom(
        mol, config['package_name'], config.cp2k_general_settings['basis'], config['path_hdf5'])

    index = 0
    for i in range(n_atoms):
        q[i, :, :] = np.dot(
            c_mo[index:(index + n_sph_atoms[i]), :].T,
            c_mo[index:(index + n_sph_atoms[i]), :],
        )
        index += n_sph_atoms[i]

    return q


def compute_MNOK_integrals(mol, xc_dft):
    """TODO: add Documentation."""
    n_atoms = len(mol)
    r_ab = get_r_ab(mol)
    hardness_vec = np.stack([hardness(m[0]) for m in mol]).reshape(n_atoms, 1)
    hard = np.add(hardness_vec, hardness_vec.T)
    beta = xc(xc_dft)['beta1'] + xc(xc_dft)['ax'] * xc(xc_dft)['beta2']
    alpha = xc(xc_dft)['alpha1'] + xc(xc_dft)['ax'] * xc(xc_dft)['alpha2']
    if (xc(xc_dft)['type'] == 'pure'):
        gamma_J = np.zeros((n_atoms, n_atoms))
    else:
        gamma_J = np.power(
            1 / (np.power(r_ab, beta) + np.power((xc(xc_dft)['ax'] * hard), -beta)),
            1 / beta,
        )
    gamma_K = np.power(
        1 / (np.power(r_ab, alpha) + np.power(hard, -alpha)),
        1 / alpha,
    )

    return gamma_J, gamma_K


def construct_A_matrix_tddft(pqrs_J, pqrs_K, nocc, nvirt, xc_dft, e):
    """TODO: add Documentation."""
    # This is the exchange integral entering the A matrix.
    #  It is in the format (nocc, nvirt, nocc, nvirt)
    k_iajb = pqrs_K[:nocc, nocc:, :nocc, nocc:].reshape(
        nocc * nvirt, nocc * nvirt)

    # This is the Coulomb integral entering in the A matrix.
    # It is in the format: (nocc, nocc, nvirt, nvirt)
    k_ijab_tmp = pqrs_J[:nocc, :nocc, nocc:, nocc:]

    # To get the correct order in the A matrix, i.e. (nocc, nvirt, nocc, nvirt),
    # we have to swap axes
    k_ijab = np.swapaxes(k_ijab_tmp, axis1=1, axis2=2).reshape(
        nocc * nvirt, nocc * nvirt)

    # They are in the m x m format where m is the number of excitations = nocc * nvirt
    a_mat = 2 * k_iajb - k_ijab

    # Generate a vector with all possible ea - ei energy differences
    e_diff = -np.subtract(
        e[:nocc].reshape(nocc, 1), e[nocc:].reshape(nvirt, 1).T).reshape(nocc * nvirt)
    np.fill_diagonal(a_mat, np.diag(a_mat) + e_diff)

    return a_mat
