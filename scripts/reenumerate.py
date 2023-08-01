#!/usr/bin/env python
"""Comman line interface to reenumerate a bunch of HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Iterable

import h5py

from nanoqm import logger
from nanoqm.workflows.distribute_jobs import compute_number_of_geometries

msg = "reenumerate.py -n name_project -d directory"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-n', required=True,
                    help="Name of the project")
parser.add_argument('-d', help="work directory", default='.')


def create_new_group_names(groups: Iterable[str], index: int) -> list[str]:
    """Create new names using index for groups."""
    new_names = []
    for g in groups:
        root, number = g.split('_')
        new_index = index + int(number)
        new_names.append(f"{root}_{new_index}")
    return new_names


def rename_groups_in_hdf5(path_hdf5: Path, project: str, index: int) -> None:
    """Rename the group inside project using index."""
    with h5py.File(path_hdf5, 'r+') as f5:
        groups = list(f5[project].keys())
        new_names = create_new_group_names(groups, index)
        root = f5[project]
        # Move all the groups to some intermediate names
        # to avoid collisions
        for old, new in zip(groups, new_names):
            if old != new:
                root.move(old, f"000_{new}")

        # Finally rename everything
        for old, new in zip(groups, new_names):
            if old != new:
                root.move(f"000_{new}", new)


def reenumerate(project: str, folder_and_hdf5: tuple[str, str], acc: int) -> int:
    """Reenumerate hdf5 files in folder using acc."""
    folder, hdf5 = folder_and_hdf5
    logger.info(f"Renaming {hdf5} by adding {acc} to the index")
    rename_groups_in_hdf5(Path(hdf5), project, acc)

    # Count the number of geometries in the chunk
    path_to_trajectory = next(Path(folder).glob("chunk_xyz_*"))
    number_of_geometries = compute_number_of_geometries(path_to_trajectory)
    return acc + number_of_geometries


def main() -> None:
    """Parse the command line arguments and run workflow."""
    args = parser.parse_args()
    project = args.n
    directory = Path(args.d)

    # Get the folders where the trajectories are stored
    folders = [x for x in directory.glob("chunk_*") if x.is_dir()]
    folders.sort()

    # Get the hdf5 files
    hdf5_files = list(directory.glob("*.hdf5"))
    hdf5_files.sort()

    acc = 0
    for folder_and_hdf5 in zip(folders, hdf5_files):
        acc = reenumerate(project, folder_and_hdf5, acc)


if __name__ == "__main__":
    main()
