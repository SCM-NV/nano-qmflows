#!/usr/bin/env python

"""Convert old HDF5 files to the new storage layout."""

import argparse
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional, Set

import h5py
import numpy as np


def exists(input_file: str) -> Path:
    """Check if the input file exists."""
    path = Path(input_file)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{input_file} doesn't exist!")

    return path


class LegacyConverter:
    """Convert legacy HDF5 files to the new storage layout."""

    def __init__(self, source: h5py.File, dest: h5py.File) -> None:
        """Initialize the converter."""
        self.source = source
        self.dest = dest
        self.project = self.get_project_name()

    def get_project_name(self) -> str:
        """Get the project root name."""
        # There are only two names onder root: cp2k and project name
        diff = set(self.source.keys()) - {'cp2k'}
        return diff.pop()

    def copy_data_set(self, old_path: str, new_path: str) -> None:
        """Copy data set from old ``source`` to new ``dest``."""
        if old_path in self.source:
            data = self.source[old_path][()]
            self.dest.require_dataset(new_path, shape=np.shape(data), data=data, dtype=np.float32)

    def copy_node_values(self, old_names: Iterable[str], new_names: Iterable[str]) -> None:
        """Copy the data set values from the old file to the new one."""
        for old, new in zip(old_names, new_names):
            self.copy_data_set(old, new)

    def copy_orbitals(self) -> None:
        """Copy orbitals from old ``source`` to new ``dest``."""
        points = [k for k in self.source[self.project].keys() if k.startswith("point_")]
        keys = {"coefficients", "eigenvalues", "energy"}
        # "project/point_x/cp2k/mo/<coefficients,eigenvalues,energy>"
        old_names = chain(*[[f"{self.project}/{point}/cp2k/mo/{k}" for point in points] for k in keys])
        new_names = chain(*[[f"{k}/{point}" for point in points] for k in keys])
        self.copy_node_values(old_names, new_names)

    def copy_couplings(self) -> None:
        """Copy couplings and swap matrix."""
        couplings = [k for k in self.source[self.project].keys() if k.startswith("coupling_")]
        old_names = [f"{self.project}/{cs}" for cs in couplings]
        new_names = couplings
        self.copy_node_values(old_names, new_names)

        swaps = f"{self.project}/swaps"
        old_names = [swaps]
        new_names = ["swaps"]
        self.copy_node_values(old_names, new_names)

    def copy_overlaps(self) -> None:
        """Copy the overlaps to the new layout."""
        overlaps = [k for k in self.source[self.project].keys() if k.startswith("overlaps_")]
        keys = {"mtx_sji_t0", "mtx_sji_t0_corrected"}
        old_names = chain(*[[f"{self.project}/{over}/{k}" for over in overlaps] for k in keys])
        new_names = chain(*[[f"{over}/{k}" for over in overlaps] for k in keys])
        self.copy_node_values(old_names, new_names)

    def copy_multipoles(self) -> None:
        """Copy the multipoles to the new layout."""
        multipole = f"{self.project}/multipole"
        if multipole in self.source:
            points = [k for k in self.source[multipole].keys() if k.startswith("point_")]
            old_names = [f"{self.project}/multipole/{p}/dipole" for p in points]
            new_names = [f"dipole/{p}" for p in points]
            self.copy_node_values(old_names, new_names)

    def copy_all(self) -> None:
        """Copy all the has been stored in the legacy format."""
        for fun in {"copy_orbitals", "copy_couplings", "copy_overlaps", "copy_multipoles"}:
            method = getattr(self, fun)
            method()


def convert(path_hdf5: Path) -> None:
    """Convert ``path_hdf5`` to new storage format."""
    new_hdf5 = path_hdf5
    old_hdf5 = path_hdf5.rename(f'old_{path_hdf5.name}')

    with h5py.File(new_hdf5, 'a') as dest, h5py.File(old_hdf5, 'r') as source:
        converter = LegacyConverter(source, dest)
        converter.copy_all()


def main():
    """Perform the conversion."""
    parser = argparse.ArgumentParser("convert_legacy_hdf5")
    parser.add_argument("input", type=exists, help="HDF5 file to convert")

    args = parser.parse_args()
    convert(args.input)


if __name__ == "__main__":
    main()
