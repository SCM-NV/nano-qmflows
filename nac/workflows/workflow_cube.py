__author__ = "Felipe Zapata"


# ================> Python Standard  and third-party <==========
from collections import namedtuple
from typing import  (Dict, List)
import logging

GridCube = namedtuple("GridCube", ("center", "voxels", "vectors", "grid"))


def main(package_name: str, project_name: str, package_args: Dict,
         guess_args: Dict=None, geometries: List=None, dictCGFs: Dict=None,
         calc_new_wf_guess_on_points: str=None, path_hdf5: str=None,
         enumerate_from: int=0, package_config: Dict=None, traj_folders: List=None,
         work_dir: str=None, basisname: str=None, hdf5_trans_mtx: str=None,
         nHOMO: int=None, algorithm='levine', ignore_warnings=False) -> None:
    pass
    # # Start logging event
    # file_log = '{}.log'.format(project_name)
    # logging.basicConfig(filename=file_log, level=logging.DEBUG,
    #                     format='%(levelname)s:%(message)s  %(asctime)s\n',
    #                     datefmt='%m/%d/%Y %I:%M:%S %p')

    # # prepare Cp2k Jobs
    # # Point calculations Using CP2K
    # mo_paths_hdf5 = calculate_mos(
    #     package_name, geometries, project_name, path_hdf5, traj_folders,
    #     package_args, guess_args, calc_new_wf_guess_on_points,
    #     enumerate_from, package_config=package_config,
    #     ignore_warnings=ignore_warnings)

