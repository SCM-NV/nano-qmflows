"""Module to store data in the HDF5."""

__all__ = ['StoreasHDF5', 'store_cp2k_basis']

from os.path import join

import numpy as np

from qmflows.parsers.cp2KParser import readCp2KBasis


class StoreasHDF5:
    """Class to store inside a HDF5 file numerical array with optional attributes."""

    def __init__(self, file_h5):
        self.file_h5 = file_h5

    def save_data(self, path_property, data):
        """Create a data set using ``data`` and saves the data using `pathProperty` in the HDF5 file.

        :param pathProperty: path to store the property in HDF5.
        :type pathProperty: String
        :param data: Numeric array containing the property.
        :type data: Numpy array
        :returns: h5py.Dataset
        """
        return self.file_h5.require_dataset(path_property, shape=np.shape(data),
                                            data=data, dtype=np.float32)

    def save_data_attrs(self, nameAttr, attr, pathProperty, data):
        """Create a data set using ``data`` and some attributes.

        :param nameAttr: Name of the attribute assoaciated with the data.
        :type nameAttr: String
        :param attr: Actual atttribute.
        :type attr: String | Numpy array
        :param pathProperty: path to store the property in HDF5.
        :type pathProperty: String
        :param data: Numeric array containing the property.
        :type data: Numpy array
        :returns: None
        """

        dset = self.save_data(pathProperty, data)
        dset.attrs[nameAttr] = attr

    def saveBasis(self, parserFun, pathBasis):
        """Store the basis set.

        :param parserFun: Function to parse the file containing the
                          information about the primitive contracted Gauss
                          functions.
        :param pathBasis: Absolute path to the file containing the basis
                          sets information.
        :type pathBasis: String.
        :returns: None
        """
        keys, vals = parserFun(pathBasis)
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


def store_cp2k_basis(file_h5, key):
    """Read the CP2K basis set into an HDF5 file."""
    storeCp2k = StoreasHDF5(file_h5)

    return storeCp2k.saveBasis(readCp2KBasis, *key.args)
