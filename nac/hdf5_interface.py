"""Module to store data in the HDF5."""

import h5py
import numpy as np


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
