\
from nac import initialize
from nac.common import (change_mol_units, retrieve_hdf5_data)
from nac.schedule.components import calculate_mos
from nac.integrals.nonAdiabaticCoupling import calcOverlapMtx
from qmworks import run, Settings, templates
from qmworks.parsers import parse_string_xyz
from scipy import sparse

import numpy as np

def main():

    scratch_path = 'path/where/the/CP2K/calculations/are/done'
    project_name = 'project_name'
    path_traj_xyz = 'path/to/file/xyz'
    basisname     = 'DZVP-MOLOPT-SR-GTH'
    path_hdf5 = 'phase_references.hdf5'
    path_basis = 'path_to_cp2k_basis_file'
    path_potential = 'path_to_cp2k_pot_file'
    cell_parameters = None
    cell_angles = [90] * 3
    added_mos = 20
    # Range of orbitals printed by CP2K
    range_orbitals = None, None

    # Coupling parameters
    nHOMO = None
    couplings_range = None, None 

    # CP2K Settings
    cp2k_args = templates.singlepoint
    cp2k_args.basis = basisname
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = cell_parameters
    cp2k_args.cell_angles = cell_angles
    main_dft = cp2k_args.specific.cp2k.force_eval.dft
    main_dft.scf.added_mos = added_mos
    main_dft.scf.eps_scf = 1e-04
    main_dft['print']['mo']['mo_index_range'] = "{} {}".format(*range_orbitals)
    cp2k_args.specific.cp2k.force_eval.subsys.cell.periodic = 'None'


    cp2k_OT_args = cp2k_args.copy()
    ot_dft = cp2k_OT_args.specific.cp2k.force_eval.dft
    ot_dft.scf.scf_guess = 'atomic'
    ot_dft.scf.ot.minimizer = 'DIIS'
    ot_dft.scf.ot.n_diis = 7
    ot_dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    ot_dft.scf.added_mos = 0
    ot_dft.scf.eps_scf = 1e-04

    
    # End of user serviceable Code
    initial = initialize(project_name, path_traj_xyz, basisname,
                         path_hdf5=path_hdf5,
                         scratch_path=scratch_path, path_basis=path_basis,
                         path_potential=path_potential)

    arr = workflow_phase_references(project_name, cp2k_args, cp2k_OT_args,
                                    nHOMO=nHOMO,
                                    couplings_range=couplings_range,
                                    scratch_path=scratch_path,
                                    **initial)

    np.save('references', arr)

    print("The phase reference array has been stored at: ", "references.npy")


def workflow_phase_references(project_name, cp2k_args, cp2k_OT_args=None,
                              nHOMO=None, couplings_range=None,
                              geometries=None,
                              path_hdf5=None, traj_folders=None,
                              calc_new_wf_guess_on_points=None,
                              enumerate_from=None, package_config=None,
                              dictCGFs=None, hdf5_trans_mtx=None,
                              scratch_path=None, **kwargs):
    """
    Worflow to compute the references matrix containing the phase signs
    use to keep constant the phase during the computation of the
    adiabatic couplings.
    """
    # compute Molecular orbitals of the first two points
    promises = calculate_mos("cp2k", geometries, project_name,
                             path_hdf5, traj_folders, cp2k_args,
                             cp2k_OT_args, calc_new_wf_guess_on_points,
                             enumerate_from=0,
                             package_config=package_config)

    mo_paths = run(promises, folder=scratch_path)
    # read the matrix from the HDF5 and transform it to
    # sparse representation
    trans_mtx = retrieve_hdf5_data(path_hdf5, hdf5_trans_mtx)
    trans_mtx = sparse.csr_matrix(trans_mtx)

    # Read the molecular geometries
    mol0, mol1 = tuple(map(parse_string_xyz, geometries))

    # Dimension of the square overlap atomic matrix
    dim = sum(len(dictCGFs[at.symbol]) for at in mol0)

    # Compute the atomic overlap 
    Suv = calcOverlapMtx(dictCGFs, dim, mol0, mol1)

    # extract the molecular orbital coefficients
    css_0, css_1 = tuple(map(lambda j:
                             retrieve_hdf5_data(path_hdf5,
                                                mo_paths[j][1]), range(2)))

    # Take a subset of orbitas if the user request them
    if all(x is not None for x in [nHOMO, couplings_range]):
        lowest = couplings_range[0]
        highest = nHOMO + couplings_range[1]
        css_0 = css_0[:, lowest, highest]
        css_1 = css_1[:, lowest, highest]

    # Compute the overlap in sphericals
    transpose = trans_mtx.transpose()
    css_0T = np.transpose(css_0)
    Suv = trans_mtx.dot(sparse.csr_matrix.dot(Suv, transpose))
    references = np.dot(css_0T, np.dot(Suv, css_1))

    return np.sign(references)

    
if __name__ == "__main__":
    main()

