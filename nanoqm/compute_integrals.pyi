import numpy as np
import numpy.typing as npt

def compute_integrals_couplings(
    __path_xyz_1: str,
    __path_xyz_2: str,
    __path_hdf5: str,
    __basis_name: str,
) -> npt.NDArray[np.float64]: ...

def compute_integrals_multipole(
    __path_xyz_1: str,
    __path_xyz_2: str,
    __basis_name: str,
    __multipole: str,
) -> npt.NDArray[np.float64]: ...
