"""Module to store data in the HDF5."""

__all__ = ['store_cp2k_basis']
from os.path import join

import h5py
import numpy as np

from qmflows.parsers.cp2KParser import readCp2KBasis


class StoreasHDF5:
    """Class to store inside a HDF5 file numerical array with optional attributes."""

    def __init__(self, file_h5):
        """Save the instance of the h5py File."""
        self.file_h5 = file_h5

    def save_data(self, path_property: str, data: np.array) -> h5py.Dataset:
        """Create a data set using ``data`` and saves the data using `path_property` in the HDF5 file.

        :param pathProperty: path to store the property in HDF5.
        :param data: Numeric array containing the property.
        """
        return self.file_h5.require_dataset(path_property, shape=np.shape(data),
                                            data=data, dtype=np.float32)

    def save_data_attrs(self, name_attr: str, attr: str, path_property: str, data: np.array):
        """Create a data set using ``data`` and some attributes.

        :param name_attr: Name of the attribute assoaciated with the data.
        :param attr: Actual atttribute.
        :param pathProperty: path to store the property in HDF5.
        :param data: Numeric array containing the property.
        :returns: None
        """
        dset = self.save_data(path_property, data)
        dset.attrs[name_attr] = attr

    def saveBasis(self, parser_fun: callable, path_basis: str):
        """Store the basis set.

        :param parser_fun: Function to parse the file containing the
                          information about the primitive contracted Gauss
                          functions.
        :param path_basis: Absolute path to the file containing the basis
                          sets information.
        :returns: None
        """
        keys, vals = parser_fun(path_basis)
        pathsExpo = [join("cp2k/basis", xs.atom, xs.basis, "exponents")
                     for xs in keys]
        pathsCoeff = [join("cp2k/basis", xs.atom, xs.basis,
                           "coefficients") for xs in keys]

        for ps, es in zip(pathsExpo, [xs.exponents for xs in vals]):
            self.save_data(ps, es)

        fss = [xs.basisFormat for xs in keys]
        css = [xs.coefficients for xs in vals]

        # save basis set coefficients and their correspoding format
        for path, fs, css in zip(pathsCoeff, fss, css):
            self.save_data_attrs("basisFormat", str(fs), path, css)


def store_cp2k_basis(file_h5, path_basis):
    """Read the CP2K basis set into an HDF5 file."""
    storeCp2k = StoreasHDF5(file_h5)

    return storeCp2k.saveBasis(readCp2KBasis, path_basis)
