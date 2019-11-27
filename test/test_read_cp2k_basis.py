"""Read basis in CP2K format."""
import os

import h5py
import pkg_resources

from nac.workflows.initialization import store_cp2k_basis


def test_read_cp2k_basis(tmp_path):
    """Read Basis set in CP2K format."""
    tmp_hdf5 = os.path.join(tmp_path, 'cp2k_basis.hdf5')

    path_basis = pkg_resources.resource_filename(
        "nac", "basis/BASIS_MOLOPT")

    coefficients_format_carbon_DZVP_MOLOPT_GTH = '[2, 0, 2, 7, 2, 2, 1]'
    with h5py.File(tmp_hdf5, 'a') as f5:
        store_cp2k_basis(f5, path_basis)

        dset = f5["cp2k/basis/c/DZVP-MOLOPT-GTH/coefficients"]
        # Check that the format is store
        assert dset.attrs['basisFormat'] == coefficients_format_carbon_DZVP_MOLOPT_GTH
        # Check Shape of the coefficients
        assert dset.shape == (5, 7)
