
from nac.basisSet import (compute_normalization_sphericals, create_dict_CGFs)
from nac.common import (change_mol_units, store_arrays_in_hdf5)
from nac.schedule.components import split_file_geometries
from nac.integrals import calc_transf_matrix
from nac.workflows.worflow_cube import (GridCube, workflow_compute_cubes)

from os.path import join
from qmworks.parsers import readXYZ
import h5py
# import numpy as np
import os
import shutil

scratch_path = 'scratch'
path_original_hdf5 = 'test/test_files/ethylene.hdf5'
path_test_hdf5 = join(scratch_path, 'test.hdf5')
path_xyz = 'test/test_files/ethylene.xyz'
project_name = 'ethylene'
package_args = None


def fixme_test_cube():
    """
    Test the density compute to create a cube file.
    """
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # Overlap matrix in cartesian coordinates
    basisname = "DZVP-MOLOPT-SR-GTH"

    # Read coordinates
    molecule = change_mol_units(readXYZ(path_xyz))

    # String representation of the molecule
    geometries = split_file_geometries(path_xyz)

    try:
        shutil.copy(path_original_hdf5, path_test_hdf5)
        # Contracted Gauss functions
        dictCGFs = create_dict_CGFs(path_test_hdf5, basisname, molecule)

        dict_global_norms = compute_normalization_sphericals(dictCGFs)

        # Compute the transformation matrix and store it in the HDF5
        with h5py.File(path_test_hdf5, 'r') as f5:
            transf_mtx = calc_transf_matrix(
                f5, molecule, basisname, dict_global_norms, 'cp2k')

        path_transf_mtx = join(project_name, 'trans_mtx')
        store_arrays_in_hdf5(
            path_test_hdf5, path_transf_mtx, transf_mtx)

        # voxel size and Number of steps
        grid_data = GridCube(0.300939, 80)

        rs = workflow_compute_cubes(
            'cp2k', project_name, package_args, path_time_coeffs=None,
            grid_data=grid_data, guess_args=None, geometries=geometries,
            dictCGFs=dictCGFs, calc_new_wf_guess_on_points=[],
            path_hdf5=path_test_hdf5, enumerate_from=0, package_config=None,
            traj_folders=None, work_dir=scratch_path, basisname=basisname,
            hdf5_trans_mtxstr=path_transf_mtx, nHOMO=None,
            ignore_warnings=False)

        print(rs)

    finally:
        # Remove intermediate results
        shutil.rmtree(scratch_path)


if __name__ == "__main__":
    test_cube()
