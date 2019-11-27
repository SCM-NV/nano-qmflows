import h5py
import numpy as np
from nac.common import calc_orbital_Slabels, read_basis_format


def number_spherical_functions_per_atom(mol, package_name, basis_name, path_hdf5):
    """Compute the number of spherical shells per atom."""
    with h5py.File(path_hdf5, 'r') as f5:
        xs = [f5[f'{package_name}/basis/{atom[0]}/{basis_name}/coefficients']
              for atom in mol]
        ys = [calc_orbital_Slabels(
            package_name, read_basis_format(
                package_name, path.attrs['basisFormat'])) for path in xs]

        return np.stack([sum(len(x) for x in ys[i]) for i in range(len(mol))])
