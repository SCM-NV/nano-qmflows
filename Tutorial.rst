Tutorial on QMFlows-NAMD calculations (YML input)
====================

This tutorial focus on how to perform QMFlows-NAMD calculations with the newest version of QMFlows, where the *input_test_distribute_derivative_couplings.yml* input is used instead of the *distribute_job.py* input used in older versions. When using this tutorial, ensure you have the latest version of QMFlows and QMFlows-NAMD installed.

Preparing the input
--------------------
The following is an example of the `distribution input`:

.. code-block:: yaml

		workflow:
      distribute_derivative_couplings

    project_name: Cd33Se33
    runner: multiprocessing
    dt: 1
    active_space: [10, 10]
    algorithm: "levine"
    tracking: False
    path_hdf5: "test/test_files/Cd33Se33.hdf5"
    path_traj_xyz: "test/test_files/Cd33Se33_fivePoints.xyz" 
    scratch_path: "/tmp/namd"
    workdir: "."
    blocks: 5

    job_scheduler:
      scheduler: SLURM
      nodes: 1
      tasks: 24
      wall_time: "24:00:00"
      load_modules: "source activate qmflows\nmodule load cp2k/3.0"

      
    cp2k_general_settings:
      basis:  "DZVP-MOLOPT-SR-GTH"
      path_basis: "test/test_files/BASIS_MOLOPT"
      path_potential: "test/test_files/GTH_POTENTIALS"
      potential: "GTH-PBE"
      cell_parameters: 28.0
      cell_angles: [90.0, 90.0, 90.0]

      cp2k_settings_main:
        specific:
          template: pbe_main

      cp2k_settings_guess:
        template:
          pbe_guess


The previous input can be found at input_test_distribute_derivative_couplings.yml_. Copy this file to a folder where you want start the QMFlows calculations. 

The *input_test_distribute_derivative_couplings.yml* file contains all settings to perform the calculations and needs to be edited according to your system and preferences. Pay attention to the following parameters: *workdir, blocks, job_scheduler, basis_name, project_name, nHOMO, mo_index_range, path_basis, path_potential, path_hdf5, path_traj_xyz, scratch_path*. 

- **workdir**: This is the location where the log and the results will be written. Default setting is current directory.
- **blocks**: Number of blocks (chunks) is related to how the trajectory is split up. As our typical trajectories are quite large (+- 2000 structures), it is convenient to split the trajectory up into multiple chunks so that several calculations can be performed simultaneously. Generally around 4-5 blocks is sufficient, depending on the length of your trajectory and the size of your system. 
- **job_scheduler**: The launch setting for the calculations can be set with the job_scheduler. 
- **basis_name**: (Why specify this when you have to specify it for each atom type?)
- **project_name**: Project name for the calculations. 
- **activate_space**: Range of `(occupied, virtual)` molecular orbitals to computed the derivate couplings.
- **path_basis, path_potential**: Path to the CP2K basis set and potentials files. 
- **path_hdf5**: Path where the hdf5 should be created / can be found.
- **path_traj_xyz**: Path to the full trajectory.
- **scratch_path**: A scratch path is required to perform the calculations. For large systems, the .hdf5 files can become quite large (in GBs) and calculations are instead performed in the scratch workspace. The final results will also be stored here.

The settings below these initial parameters are the settings used to generate the cp2k input. You can customize these settings as required for your calculations. Use the cp2k manual_ to create your custom input. 

.. _manual: https://manual.cp2k.org/
.. _input_test_distribute_derivative_couplings.yml: https://github.com/SCM-NV/qmflows-namd/blob/master/test/test_files/input_test_distribute_derivative_couplings.yml

Setting up the calculation 
---------------------------

Once all settings in *input_test_distribute_derivative_couplings.yml* have been customized, you will need to create the different chunks. 
  
- First, activate QMFlows:

  ``conda activate qmflows``  

- Use the command *distribute_jobs.py* to create the different chunks:

  ``distribute_jobs.py -i input_test_distribute_derivative_couplings.yml``

A number of new folders are created. In each folder you will find a launch.sh file, a chunk_xyz file and an input.yml file. In the input.yml file, you will find all your settings. Check for any possible manual errors.

- If you are satisfied with the inputs, submit each of your jobs for calculation.

You can keep track of the calculations by going to your scratch path. The location where all points of the chunks are calculated is your assigned scratch path plus project name plus a number. 

The overlaps and couplings between each state will be calculated once the single point calculations are finished. The progress can be tracked with the .log file in your working directory folders. The calculated couplings are meaningless at this point and need to be removed and recalculated, more on that later.  

Merging the chunks and recalculating the couplings 
---------------------------------------------------

Once the overlaps and couplings are all calculated, you need to merge the different chunks into a single chunk, as the overlaps between the different chunks still need to be calculated. For this you will use the *mergeHDF5.py* command that you will have if you have installed QMFlows correctly. 

You are free to choose your own HDF5 file name but for this tutorial we will use *chunk_01234.hdf5* as an example. 

- Merge the different chunk into a single file using the *mergeHDF5.py* script:

  ``mergeHDF5.py -i chunk_0.hdf5 chunk_1.hdf5 chunk_2.hdf5 chunk_3.hdf5 chunk_4.hdf5 -o chunk_01234.hdf5``

Follow -i with the names of different chunks you want to merge and follow -o the name of the merged HDF5 file.  

- Remove the couplings from the chunk_01234.hdf5 using the *removeHDF5folders.py* script. To run the script, use: 

  ``removeHDF5folders.py -pn PROJECTNAME -HDF5 chunk_01234.hdf5``

Replace PROJECTNAME with your project name. 

Using the script in this manner will only allow the couplings to be removed. 

.. Note::
   If required, you can remove all overlaps by by adding -o at the end of the previous command:

  ``removeHDF5folders.py -pn PROJECTNAME -hdf5 chunk_01234.hdf5 –o``


- Create a new subfolder in your original working directory and copy the *input.yml* file that was created for chunk 0 (when running the *distribute_jobs.py* script) to this folder. 

- Edit the *input.yml* file to include the path to the merged .hdf5, the full MD trajectory, and a new scratch path for the merged hdf5 calculations.

- Relaunch the calculation.

Once the remaining overlaps and the couplings have been calculated successfully, the hdf5 files and hamiltonians will be written to both the working directory as well as the scratch folder. The overlaps can be found in the working directory.
