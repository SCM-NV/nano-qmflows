
# from nac.basisSet import (compute_normalization_sphericals, create_dict_CGFs)
# from nac.worflows.worflow_cube import workflow_compute_cubes
# import h5py
# import numpy as np
# import os
# import shutil

# scratch_path = 'scratch'
# path_original_hdf5 = 'test/test_files/ethylene.hdf5'
# path_test_hdf5 = join(scratch_path, 'test.hdf5')


# def test_cube():
#     """
#     Test the density compute to create a cube file.
#     """
#     if not os.path.exists(scratch_path):
#         os.makedirs(scratch_path)

#     # Overlap matrix in cartesian coordinates
#     basisname = "DZVP-MOLOPT-SR-GTH"

#     try:
#         dictCGFs = create_dict_CGFs(path_hdf5, basisname, atoms)
#         shutil.copy(path_original_hdf5, path_test_hdf5)


#     finally:
#         shutil.rmtree(scratch_path)


# if __name__ == "__main__":
#     test_cube()
