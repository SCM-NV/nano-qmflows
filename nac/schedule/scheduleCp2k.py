__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from noodles import schedule  # Workflow Engine
from os.path import join

import fnmatch
import os
import plams

# ==================> Internal modules <==========
from qmworks import templates
from qmworks.packages import cp2k

# ==============================> Schedule Tasks <=========================


def prepare_cp2k_settings(geometry, files, cp2k_args, k, work_dir,
                          wfn_restart_job, store_in_hdf5, cp2k_config):
    """
    Fills in the parameters for running a single job in CP2K.

    :param geometry: Molecular geometry stored as String
    :type geometry: String
    :param files: Tuple containing the IO files to run the calculations
    :type files: nameTuple
    :parameter dict_input: Dictionary contaning the data to
    fill in the template
    :type  dict_input: Dict
    :param k: nth Job
    :type k: Int
    :parameter work_dir: Name of the Working folder
    :type      work_dir: String
    :param wfn_restart_job: Path to *.wfn cp2k file use as restart file.
    :type wfn_restart_job: String
    :param store_in_hdf5: Wether or not the numerical result are store in HDF5.
    numerical results.
    :type store_in_hdf5: Bool
    :param cp2k_config:  Parameters required by cp2k.
    :type cp2k_config: Dict
   :returns: ~qmworks.Settings
    """
    # Search for the environmental variable BASISCP2K containing the path
    # to the Basis set folder

    basis_file = cp2k_config["basis"]
    potential_file = cp2k_config["potential"]

    cp2k_args.specific.cp2k.force_eval.dft.basis_set_file_name = basis_file
    cp2k_args.specific.cp2k.force_eval.dft.potential_file_name = potential_file
    cp2k_args.specific.cp2k.force_eval.dft['print']['mo']['filename'] = files.get_MO
    cp2k_args.specific.cp2k.force_eval.subsys.topology.coord_file_name = files.get_xyz
    cp2k_args.specific.cp2k['global']['project'] = 'point_{}'.format(k)

    if wfn_restart_job is not None:
        output_dir = getattr(wfn_restart_job.archive['plams_dir'], 'path')
        xs = os.listdir(output_dir)
        wfn_file = list(filter(lambda x: fnmatch.fnmatch(x, '*wfn'), xs))[0]
        file_path = join(output_dir, wfn_file)
        cp2k_args.specific.cp2k.force_eval.dft.wfn_restart_file_name = file_path

    with open(files.get_xyz, 'w') as f:
        f.write(geometry)

    input_args = templates.singlepoint.overlay(cp2k_args)

    # Do not print the MO if they are not going to be stored in HDF5
    if not store_in_hdf5:
        del(input_args.specific.cp2k.force_eval.dft['print'])

    return input_args


@schedule
def prepare_job_cp2k(geometry, files, dict_input, k, work_dir,
                     project_name=None, hdf5_file=None, wfn_restart_job=None,
                     store_in_hdf5=True, nHOMOS=None, nLUMOS=None,
                     package_config=None):
    """
    Fills in the parameters for running a single job in CP2K.

    :param geometry: Molecular geometry stored as String
    :type geometry: String
    :param files: Tuple containing the IO files to run the calculations
    :type files: nameTuple
    :parameter dict_input: Dictionary contaning the data to
    fill in the template
    :type      dict_input: Dict
    :param k: nth Job
    :type k: Int
    :parameter work_dir: Name of the Working folder
    :type      work_dir: String
    :param hdf5_file: Path to the HDF5 file that contains the
    numerical results.
    :type hdf5_file: String
    :param wfn_restart_job: Path to *.wfn cp2k file use as restart file.
    :type wfn_restart_job: String
    :param nHOMOS: number of HOMOS to store in HDF5.
    :type nHOMOS: Int
    :param nLUMOS: number of HOMOS to store in HDF5.
    :type nLUMOS: Int
    :returns: ~qmworks.CP2K
    """
    job_settings = prepare_cp2k_settings(geometry, files, dict_input, k, work_dir,
                                         wfn_restart_job, store_in_hdf5,
                                         package_config)
    project_name = project_name if project_name is not None else work_dir

    return cp2k(job_settings, plams.Molecule(files.get_xyz), work_dir=work_dir,
                project_name=project_name, hdf5_file=hdf5_file,
                input_file_name=files.get_inp,
                out_file_name=files.get_out, store_in_hdf5=store_in_hdf5,
                nHOMOS=nHOMOS, nLUMOS=nLUMOS)
