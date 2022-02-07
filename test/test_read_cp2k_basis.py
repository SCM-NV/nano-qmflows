"""Read basis in CP2K format."""
from pathlib import Path

import h5py
import pkg_resources
from qmflows.type_hints import PathLike

from nanoqm.workflows.initialization import store_cp2k_basis


def test_read_cp2k_basis(tmp_path: PathLike) -> None:
    """Read Basis set in CP2K format."""
    tmp_hdf5 = Path(tmp_path) / 'cp2k_basis.hdf5'
    tmp_hdf5.touch()

    path_basis = pkg_resources.resource_filename(
        "nanoqm", "basis/BASIS_MOLOPT")

    coefficients_format_carbon_DZVP_MOLOPT_GTH = {'[2, 0, 2, 7, 2, 2, 1]', '(2, 0, 2, 7, 2, 2, 1)'}
    store_cp2k_basis(tmp_hdf5, path_basis)

    with h5py.File(tmp_hdf5, 'a') as f5:
        dset = f5["cp2k/basis/c/DZVP-MOLOPT-GTH/coefficients"]
        # Check that the format is store
        assert dset.attrs['basisFormat'] in coefficients_format_carbon_DZVP_MOLOPT_GTH
        # Check Shape of the coefficients
        assert dset.shape == (5, 7)
