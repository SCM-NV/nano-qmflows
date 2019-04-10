__all__ = ['workflow_stddft']

from nac.common import (
    DictConfig, angs2au, change_mol_units, h2ev, hardness, retrieve_hdf5_data,
    is_data_in_hdf5, number_spherical_functions_per_atom, store_arrays_in_hdf5, xc)
from nac.integrals.multipole_matrices import (
    compute_matrix_multipole, get_multipole_matrix)
from nac.schedule.components import calculate_mos
from nac.workflows.initialization import initialize
from os.path import join
from qmflows.parsers import parse_string_xyz
from qmflows import run
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import logging
import numpy as np

# Starting logger
logger = logging.getLogger(__name__)


def workflow_stddft(config: dict) -> None:
    """
    Compute the excited states using simplified TDDFT

    :param workflow_settings: Arguments to compute the oscillators see:
    `data/schemas/absorption_spectrum.json
    :returns: None
    """
    # MPI communicator
    comm = config.mpi_comm

    # Dictionary containing the general configuration
    config.update(initialize(config))

    # Single Point calculations settings using CP2K
    if comm is None or comm.Get_rank() == 0:
        mo_paths_hdf5 = run(calculate_mos(config), folder=config['workdir'])
        mol = parse_string_xyz(config.geometries[0])
        shape_multipoles = compute_shape_multipole(config, mol, 'dipole')
    else:
        mo_paths_hdf5 = None
        shape_multipoles = None

    if comm is not None:
        mo_paths_hdf5 = comm.bcast(mo_paths_hdf5, root=0)
        shape_multipoles = comm.bcast(shape_multipoles, root=0)

    # Store the shape of the array containig the multipoles
    config["shape_multipoles"] = shape_multipoles

    # Read structures
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for i, gs in enumerate(config.geometries)
                    if (i % config.stride) == 0]

    results = [compute_excited_states_tddft(
        config, mo_paths_hdf5[i],
        DictConfig({'i': i * config.stride, 'mol': mol}))
               for i, mol in enumerate(molecules_au)]

    return results


def compute_excited_states_tddft(config: dict, path_MOs: list, dict_input: dict):
    """
    Compute the excited states properties (energy and coefficients) for a given
    `mo_index_range` using the `tddft` method and `xc_dft` exchange functional.
    """
    if config.mpi_comm is None:
        copy_dict = extend_input_with_mol(dict_input)
        dict_input["multipoles"] = get_multipole_matrix(config, copy_dict, 'dipole')
        return prepare_oscillators_computation(config, dict_input, path_MOs)
    else:
        # compute the multipoles with MPI
        dict_input["multipoles"] = mpi_multipoles(config, dict_input, path_MOs)
        # Compute the rest in a single MPI worker
        if config.Get_rank() == 0:
            return prepare_oscillators_computation(config, dict_input, path_MOs)
        else:
            return None


def mpi_multipoles(config: dict, inp: dict, path_MOs: list, multipole: str = 'dipole'):
    """
    Distribute the multipole integrals in several mpi processors integrals
    """
    # MPI variables
    comm = config.mpi_comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    worker = (inp.i // config.stride) % size

    # Node inside the HDF5 where the multipole is stored
    path_multipole_hdf5 = join(config.project_name, 'multipole', 'point_{}'.format(inp.i))

    if rank == 0:
        is_multipole_available = is_data_in_hdf5(config.path_hdf5, path_multipole_hdf5)
        # Send the info to the worker except if the worker is itself
        if worker != 0:
            comm.Send(is_multipole_available, dest=worker, tag=10000)
        # Check if the multipole is done
        if is_multipole_available:
            multipoles = retrieve_hdf5_data(config.path_hdf5, path_multipole_hdf5)

    if rank != worker:
        multipoles = None
    else:
        if worker != 0:
            is_multipole_available = None
            comm.Recv(is_multipole_available, source=0, tag=10000)

        # If the multipole is presented in the HDF5 return
        if is_multipole_available:
            multipole = None
        else:
            multipoles = compute_matrix_multipole(inp.mol, config, multipole)
            if rank != 0:
                comm.Send(multipoles, dest=0, tag=inp.i)

    if rank == 0 and worker != 0:
        # Do not receive the array from the same process
        multipoles = np.empty(config.shape_multipoles, dtype=np.float64)
        comm.Recv(multipoles, source=worker, tag=inp.i)
        store_arrays_in_hdf5(config.path_hdf5, path_multipole_hdf5, multipoles)

    return multipoles


def compute_shape_multipole(config: dict, mol: list, multipole: str) -> tuple:
    """
    """
    # Shape of the multipole tensor
    basis = config.cp2k_general_settings["basis"]
    spherical_basis = number_spherical_functions_per_atom(
        mol, config['package_name'], basis, config.path_hdf5)
    if multipole == 'overlap':
        return (spherical_basis, spherical_basis)
    elif multipole == 'dipole':
        return (4, spherical_basis, spherical_basis)
    elif multipole == 'quadrupole':
        return (7, spherical_basis, spherical_basis)
    else:
        raise NotImplementedError("Multipole {} has not been implemented".format(multipole))


def extend_input_with_mol(inp: dict) -> dict:
    """
    Add molecule to the dict and return a `DictConfig` object
    """
    copy_dict = DictConfig(inp.copy())
    copy_dict["mol"] = change_mol_units(inp["mol"], factor=1/angs2au)

    return copy_dict


def prepare_oscillators_computation(config: dict, inp: dict, path_MOs: list):
    """
    Distribute the multipole integrals in the available cores
    """
    logger.info("Reading energies and mo coefficients")
    # type of calculation
    energy, c_ao = retrieve_hdf5_data(config.path_hdf5, path_MOs)

    # Dictionary with the input to compute the multipoles
    inp.update({
        "energy": energy, "c_ao": c_ao, "nocc": config.active_space[0],
        "nvirt": config.active_space[1]})

    # read data from the HDF5 or calculate it on the fly
    inp["overlap"] = inp.multipoles[0]

    # retrieve or compute the omega xia values
    omega, xia = get_omega_xia(config, inp)

    # add arrays to the dictionary
    multipoles = inp.multipoles
    inp.update({"multipoles": multipoles[1:], "omega": omega, "xia": xia})

    return compute_oscillator_strengths(
        config, inp)


def get_omega_xia(config: dict, dict_input: dict):
    """
    Search for the multipole_matrices, Omega and xia values in the HDF5,
    if they are not available compute and store them.
    """
    tddft = config.tddft.lower()

    def compute_omega_xia():
        if tddft == 'sing_orb':
            return compute_sing_orb(dict_input)
        else:
            return compute_std_aproximation(config, dict_input)

    # search data in HDF5
    root = join(config.project_name, 'omega_xia', tddft, 'point_{}'.format(dict_input.i))
    paths_omega_xia = [join(root, x) for x in ("omega", "xia")]

    if is_data_in_hdf5(config.path_hdf5, paths_omega_xia):
        return tuple(retrieve_hdf5_data(config.path_hdf5, paths_omega_xia))
    else:
        omega, xia = compute_omega_xia()
        store_arrays_in_hdf5(config.path_hdf5, paths_omega_xia[0], omega)
        store_arrays_in_hdf5(config.path_hdf5, paths_omega_xia[1], xia)

        return omega, xia


def compute_sing_orb(inp: dict):
    """
    Single Orbital approximation.
    """
    energy, nocc, nvirt = [getattr(inp, x) for x in ("energy", "nocc", "nvirt")]
    omega = -np.subtract(
        energy[:nocc].reshape(nocc, 1), energy[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt)
    xia = np.eye(nocc*nvirt)

    return omega, xia


def compute_std_aproximation(config: dict, dict_input: dict):
    """
    Compute the oscillator strenght using either the stda or stddft approximations.
    """
    logger.info("Reading or computing the dipole matrices")

    # Make a function tha returns in transition density charges
    logger.info("Computing the transition density charges")
    # multipoles[0] is the overlap matrix
    q = transition_density_charges(
        dict_input.mol, config, dict_input.overlap, dict_input.c_ao)

    # Make a function that compute the Mataga-Nishimoto-Ohno_Klopman
    # damped Columb and Excgange law functions
    logger.info("Computing the gamma functions for Exchange and Coulomb integrals")
    gamma_J, gamma_K = compute_MNOK_integrals(dict_input["mol"], config.xc_dft)

    # Compute the Couloumb and Exchange integrals
    # If xc_dft is a pure functional, ax=0, thus the pqrs_J ints are not needed
    # and can be set to 0
    logger.info("Computing the Exchange and Coulomb integrals")
    if (xc(config.xc_dft)['type'] == 'pure'):
        size = dict_input.energy.size
        pqrs_J = np.zeros((size, size, size, size))
    else:
        pqrs_J = np.tensordot(q, np.tensordot(q, gamma_J, axes=(0, 1)), axes=(0, 2))
    pqrs_K = np.tensordot(q, np.tensordot(q, gamma_K, axes=(0, 1)), axes=(0, 2))

    # Construct the Tamm-Dancoff matrix A for each pair of i->a transition
    logger.info("Constructing the A matrix for TDDFT calculation")
    a_mat = construct_A_matrix_tddft(
        pqrs_J, pqrs_K, dict_input.nocc, dict_input.nvirt, config.xc_dft, dict_input.energy)

    if config.tddft == 'stddft':
        logger.info('sTDDFT has not been implemented yet !')
        # Solve the eigenvalue problem = A * cis = omega * cis
    elif config.tddft == 'stda':
        logger.info("This is a TDA calculation ! \n Solving the eigenvalue problem")
        omega, xia = np.linalg.eig(a_mat)
    else:
        msg = "Only the stda method is available"
        raise RuntimeError(msg)

    return omega, xia


def compute_oscillator_strengths(config: dict, inp: dict):
    """
    Compute oscillator strengths
    The formula can be rearranged like this:
    f_I = 2/3 * np.sqrt(2 * omega_I) * sum_ia ( np.sqrt(e_diff_ia) * xia * tdm_x) ** 2 + y^2 + z^2

    :param i: index
    :param mol: molecular geometry
    :param tddft: type of calculation
    :param config: Setting for the current calculation
    :param energy: energy of the orbitals
    :param c_ao: coefficients of the molecular orbitals
    :param nocc: number of occupied orbitals
    :param nvirt: number of virtual orbitals
    :param omega: Omega parameter
    :param multipoles: 3D Tensor with the x,y,z components
    """
    tddft = config.tddft.lower()
    # 1) Get the inp.energy matrix i->a. Size: Inp.Nocc * Inp.Nvirt
    delta_ia = -np.subtract(
        inp.energy[:inp.nocc].reshape(inp.nocc, 1),
        inp.energy[inp.nocc:].reshape(inp.nvirt, 1).T).reshape(inp.nocc*inp.nvirt)

    def compute_transition_matrix(matrix):
        return np.stack(
            [np.sum(
                np.sqrt(2 * delta_ia / inp.omega[i]) * inp.xia[:, i] * matrix)
             for i in range(inp.nocc*inp.nvirt)])

    # 2) Compute the transition dipole matrix TDM(i->a)
    # Call the function that computes transition dipole moments integrals
    logger.info("Reading or computing the transition dipole matrix")

    def compute_tdmatrix(k):
        return np.linalg.multi_dot(
            [inp.c_ao[:, :inp.nocc].T, inp.multipoles[k, :, :],
             inp.c_ao[:, inp.nocc:]]).reshape(inp.nocc*inp.nvirt)

    td_matrices = (compute_tdmatrix(k) for k in range(3))

    # 3) Compute the transition dipole moments for each excited state i->a. Size: n_exc_states
    d_x, d_y, d_z = tuple(
        compute_transition_matrix(m) for m in td_matrices)

    # 4) Compute the oscillator strength

    if tddft == 'sing_orb':
        f = 2 / 3 * inp.omega * (td_matrices[0] ** 2 + td_matrices[1] ** 2 + td_matrices[2] ** 2)
    else:
        f = 2 / 3 * inp.omega * (d_x ** 2 + d_y ** 2 + d_z ** 2)

    # Write to output
    inp.update({"dipole": (d_x, d_y, d_z), "oscillator": f})
    write_output(config, inp)


def write_output(config: dict, inp: dict):
    """
    Write the results using numpy functionality
    """
    output = write_output_tddft(inp)

    path_output = join(config.workdir, 'output_{}_{}.txt'.format(inp.i, config.tddft))
    fmt = '{:^5s}{:^14s}{:^8s}{:^11s}{:^11s}{:^11s}{:^11s}{:<5s}{:^10s}{:<5s}{:^11s}{:^11s}'
    header = fmt.format(
        'state', 'inp.energy', 'f', 't_dip_x', 't_dip_y', 't_dip_y', 'weight',
        'from', 'inp.energy', 'to', 'inp.energy', 'delta_E')
    np.savetxt(path_output, output,
               fmt='%5d %10.3f %10.5f %10.5f %10.5f %10.5f %10.5f %3d %10.3f %3d %10.3f %10.3f',
               header=header)


def ex_descriptor(omega, f, xia, n_lowest, c_ao, s, tdm, tqm, nocc, nvirt, mol, config):
    """
    ADD DOCUMENTATION
    """
    # Reshape xia
    xia_I = xia.reshape(nocc, nvirt, nocc*nvirt)

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
    binding_en_apprx = np.stack((np.sum(xs[i, :, :]) / om[i]) for i in range(n_lowest))

    descriptors = write_output_descriptors(
        d_exc, d_exc_apprx, d_he, sigma_h, sigma_e, r_eh, binding_en_apprx, n_lowest, omega, f)

    return descriptors


def write_output_descriptors(
        d_exc, d_exc_apprx, d_he, sigma_h, sigma_e, r_eh, binding_ex_apprx, n_lowest, omega, f):
    """
    ADD Documentation
    """
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


def get_omega(d0I_ao, s, n_lowest):
    """
    ADD Documentation
    """
    return np.stack(
        np.trace(
            np.linalg.multi_dot([d0I_ao[i, :, :].T, s, d0I_ao[i, :, :], s]))
        for i in range(n_lowest))


def get_r_ab(mol):
    """
    ADD Documentation
    """
    coords = np.asarray([atom[1] for atom in mol])
    # Distance matrix between atoms A and B
    r_ab = cdist(coords, coords)
    return r_ab


def get_omega_ab(d0I_ao, s, n_lowest, mol, config):
    """
    ADD Documentation
    """
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
    """
    ADD Documentation
    """
    def compute_component_hole(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot([d0I_ao[i, :, :].T, moment[k, :, :], d0I_ao[i, :, :], s]))
            for i in range(n_lowest))

    def compute_component_electron(k):
        return np.stack(
            np.trace(
                np.linalg.multi_dot([d0I_ao[i, :, :].T, s, d0I_ao[i, :, :],  moment[k, :, :]]))
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
        raise RuntimeError("unkown option: {}".format(carrier))


# def write_output_tddft(nocc, nvirt, omega, f, d_x, d_y, d_z, xia, e):
def write_output_tddft(inp: dict):
    """ Write out as a table in plane text"""

    energy = inp.energy

    excs = [(i, a) for i in range(inp.nocc) for a in range(inp.nocc, inp.nvirt + inp.nocc)]

    output = np.empty((inp.nocc * inp.nvirt, 12))
    output[:, 0] = 0  # State number: we update it after reorder
    output[:, 1] = inp.omega * h2ev  # State energy in eV
    output[:, 2] = inp.oscillator  # Oscillator strength

    d_x, d_y, d_z = inp.dipole
    output[:, 3] = d_x  # Transition dipole moment in the x direction
    output[:, 4] = d_y  # Transition dipole moment in the y direction
    output[:, 5] = d_z  # Transition dipole moment in the z direction
    # Weight of the most important excitation
    output[:, 6] = np.hstack([np.max(inp.xia[:, i] ** 2) for i in range(inp.nocc*inp.nvirt)])

    # Find the index of this transition
    index_weight = np.hstack([
        np.where(
            inp.xia[:, i] ** 2 == np.max(
                inp.xia[:, i] ** 2))
        for i in range(inp.nocc * inp.nvirt)]).reshape(inp.nocc*inp.nvirt)

    # Index of the hole for the most important excitation
    output[:, 7] = np.stack([excs[index_weight[i]][0] for i in range(inp.nocc*inp.nvirt)]) + 1
    # These are the energies of the hole for the transition with the larger weight
    output[:, 8] = energy[output[:, 7].astype(int) - 1] * h2ev
    # Index of the electron for the most important excitation
    output[:, 9] = np.stack([excs[index_weight[i]][1] for i in range(inp.nocc*inp.nvirt)]) + 1
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
    """
    ADD Documentation
    """
    n_atoms = len(mol)
    sqrt_s = sqrtm(s)
    c_mo = np.dot(sqrt_s, c_ao)
    # Size of the transition density tensor : n_atoms x n_mos x n_mos
    q = np.zeros((n_atoms, c_mo.shape[1], c_mo.shape[1]))
    n_sph_atoms = number_spherical_functions_per_atom(
        mol, config['package_name'], config.cp2k_general_settings['basis'], config['path_hdf5'])

    index = 0
    for i in range(n_atoms):
        q[i, :, :] = np.dot(
            c_mo[index:(index + n_sph_atoms[i]), :].T, c_mo[index:(index + n_sph_atoms[i]), :])
        index += n_sph_atoms[i]

    return q


def compute_MNOK_integrals(mol, xc_dft):
    """
    ADD Documentation
    """
    n_atoms = len(mol)
    r_ab = get_r_ab(mol)
    hardness_vec = np.stack([hardness(mol[i][0]) for i in range(n_atoms)]).reshape(n_atoms, 1)
    hard = np.add(hardness_vec, hardness_vec.T)
    beta = xc(xc_dft)['beta1'] + xc(xc_dft)['ax'] * xc(xc_dft)['beta2']
    alpha = xc(xc_dft)['alpha1'] + xc(xc_dft)['ax'] * xc(xc_dft)['alpha2']
    if (xc(xc_dft)['type'] == 'pure'):
        gamma_J = np.zeros((n_atoms, n_atoms))
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
