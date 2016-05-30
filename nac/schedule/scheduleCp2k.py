__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from noodles import schedule  # Workflow Engine
from os.path import join

import fnmatch
import os
import plams

# ==================> Internal modules <==========
from qmworks import Settings, templates
from qmworks.fileFunctions import search_environ_var
from qmworks.packages import cp2k

# ==============================> Schedule Tasks <=========================


def prepare_cp2k_settings(geometry, files, cp2k_args, k, work_dir,
                          wfn_restart_job, store_in_hdf5):
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
   :returns: ~qmworks.Settings
    """
    # Search for the environmental variable BASISCP2K containing the path
    # to the Basis set folder

    basis_file = search_environ_var('BASISCP2K')
    potential_file = search_environ_var('POTENTIALCP2K')

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
        cp2k_args.specific.cp2k.force_eval.dft.scf.diagonalization.jacobi_threshold = 1e-6
    else:
        cp2k_args.specific.cp2k.force_eval.dft.scf.scf_guess = 'atomic'
        cp2k_args.specific.cp2k.force_eval.dft.scf.ot.minimizer = 'DIIS'
        cp2k_args.specific.cp2k.force_eval.dft.scf.ot.n_diis = 7
        cp2k_args.specific.cp2k.force_eval.dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
        cp2k_args.specific.cp2k.force_eval.dft.scf.added_mos = 0

    with open(files.get_xyz, 'w') as f:
        f.write(geometry)

    input_args = templates.singlepoint.overlay(cp2k_args)
    
    # Do not print the MO if they are not going to be stored in HDF5
    if not store_in_hdf5:
        del(input_args.specific.cp2k.force_eval.dft['print'])

    return input_args


@schedule
def prepare_job_cp2k(geometry, files, dict_input, k, work_dir, hdf5_file=None,
                     wfn_restart_job=None, store_in_hdf5=True):
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
    :param farming_use_guess: Use a guess for the WF using a previous job.
    :type farming_use_guess: Bool
    :returns: ~qmworks.CP2K
    """
    job_settings = prepare_cp2k_settings(geometry, files, dict_input, k, work_dir,
                                         wfn_restart_job, store_in_hdf5)

    return cp2k(job_settings, plams.Molecule(files.get_xyz), work_dir=work_dir,
                hdf5_file=hdf5_file, input_file_name=files.get_inp,
                out_file_name=files.get_out, store_in_hdf5=store_in_hdf5)


def prepare_farming_cp2k_settings(work_dirs, input_file_names, nGroups=1):
    """
    Create the input for a CP2K Farming Job, but let the Plams to take care
    of the input generation.

    :param work_dirs: List of directories containing the information to
    run a single job (e.g. coordinates).
    :type work_dirs: String List
    :param input_file_names: Names of the input files to be included in the
    farming job.
    :type input_file_names: String List
    :param nGroups: Number of Jobs to run in parallel inside the Farming.
    :type nGroups: Int
    :returns: String

    The Input for a Farming CP2K job resemble the following structure,

    &GLOBAL
       PROJECT farming
       PROGRAM FARMING
       RUN_TYPE NONE
    &END GLOBAL

    &FARMING
      GROUP_SIZE 1

      &JOB
        DIRECTORY dir-1
        INPUT_FILE_NAME job1.inp
        JOB_ID 1
      &END JOB

      &JOB
        DEPENDENCIES 1
        DIRECTORY dir-2
        INPUT_FILE_NAME job2.inp
        JOB_ID 2
      &END JOB

      ...........................
      ...........................

      &JOB
        DEPENDENCIES 1
        DIRECTORY dir-32
        INPUT_FILE_NAME job32.inp
        JOB_ID 32
      &END JOB

    &END FARMING
    """
    s = Settings()
    s.specific.cp2k['global']['project'] = 'farming'
    s.specific.cp2k['global']['program'] = 'FARMING'
    s.specific.cp2k['global']['run_type'] = 'NONE'
    s.specific.cp2k.farming.ngroups = nGroups
    s.specific.cp2k.farming.master_slave = ''

    jobs = []
    for job_id, (path, job_folder) in enumerate(zip(input_file_names,
                                                work_dirs)):
        job_name = path.split('/')[-1]
        jobs.append((job_id, job_name, job_folder))
    s.farming_jobs = jobs
    return s
