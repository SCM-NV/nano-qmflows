Derivative coupling calculation
===============================

These tutorials focus on how to compute non-adiabatic coupling vectors between molecular orbitals belonging at two different time steps, t and t+dt, of a pre-computed molecular dynamics trajectory. What this program does is to compute at each point of the trajectory, the electronic structure using DFT, and then the overlap integrals :math:`\langle \psi_{i}(t) \mid \psi_{j}(t+dt)>`. These integrals are stored and finally used to compute numerically the non-adiabatic couplings. These and the orbital energies are written in a format readable by PYXAID to perform surface hopping dynamics. 
When using this tutorial, ensure you have the latest version of QMFlows and nano-qmflows installed.

Preparing the input
--------------------
The following is an example of the inputfile for the calculation of derivative couplings for the Cd33Se33 system. The calculations are carried out with the CP2k package, which you need to have pre-installed. The level of theory is DFT/PBE. 

.. code-block:: yaml

    workflow: distribute_derivative_couplings
    project_name: Cd33Se33
    dt: 1
    active_space: [10, 10]
    algorithm: "levine"
    tracking: False
    path_hdf5: "test/test_files/Cd33Se33.hdf5"
    path_traj_xyz: "test/test_files/Cd33Se33_fivePoints.xyz" 
    scratch_path: "/tmp/namd"
    workdir: "."
    blocks: 2

    job_scheduler:
      free_format: "
      #! /bin/bash\n
      #SBATCH --job-name=Cd33Se33\n
      #SBATCH -N 1\n
      #SBATCH -t 00:15:00\n
      #SBATCH -p short\n
      source activate qmflows\n
      module load cp2k/3.0\n\n"
    
    cp2k_general_settings:
      basis:  "DZVP-MOLOPT-SR-GTH"
      potential: "GTH-PBE"
      cell_parameters: 28.0
      file_cell_parameters: "test/test_files/file_distribute_cell_parameters.txt"
      periodic: none
      executable: cp2k.popt

      cp2k_settings_main:
        specific:
          template: pbe_main

      cp2k_settings_guess:
        specific:
          template: pbe_guess


The previous input can be found at input_test_distribute_derivative_couplings.yml_. Copy this file to a folder where you want start the QMFlows calculations. 

The *input_test_distribute_derivative_couplings.yml* file contains all settings to perform the calculations and needs to be edited according to your system and preferences. Pay attention to the following parameters: *project_name, dt, active_space, algorithm, tracking, path_hdf5, path_traj_xyz, scratch_path, workdir, blocks*. 

- **project_name**: Project name for the calculations. You can choose what you wish. 
- **dt**: The size of the timestep used in your MD simulations. 
- **active_space**: Range of `(occupied, virtual)` molecular orbitals to computed the derivate couplings. For example, if 50 occupied and 100 virtual should be considered in your calculations, the active space should be set to [50, 100]. 
- **algorithm**: Algorithm to calculate derivative couplings can be set to ‘levine’ or ‘3points’.
- **tracking**: If required, you can track each state over the whole trajectory. You can also disable this option.  
- **path_hdf5**: Path where the hdf5 should be created / can be found. The hdf5 is the format used to store the molecular orbitals and other information. 
- **path_traj_xyz**: Path to the pre-computed MD trajectory. It should be provided in xyz format. 
- **scratch_path**: A scratch path is required to perform the calculations. For large systems, the .hdf5 files can become quite large (hundredths of GBs) and calculations are instead performed in the scratch workspace. The final results will also be stored here.
- **workdir**: This is the location where the logfile and the results will be written. Default setting is current directory.
- **blocks**: The number of blocks (chunks) is related to how the MD trajectory is split up. As typical trajectories are quite large (+- 5000 structures), it is convenient to split the trajectory up into multiple chunks so that several calculations can be performed simultaneously. Generally around 4-5 blocks is sufficient, depending on the length of the trajectory and the size of the system. 
- **write_overlaps**: The overlap integrals are stored locally. This option is usually activated for debugging.
- **overlaps_deph**: The overlap integrals are computed between t=0 and all othe times: <psi_i (t=0) | psi_j (t + dt)>. This option is of interest to understand how long it takes to a molecular orbital to dephase from its starting configuration. This option is disabled by default. 

The **job_scheduler** can be found below these parameters. Customize these settings according to the system and environment you are using to perform the calculations. 

In the **cp2k_general_settings**, you can customize the settings used to generate the cp2k input. You can use the cp2k manual_ to create your custom input requirements. Remember to provide a path to the folder with the cp2k basis set anc potential files.

.. _manual: https://manual.cp2k.org/
.. _input_test_distribute_derivative_couplings.yml: https://github.com/SCM-NV/nano-qmflows/blob/master/test/test_files/input_test_distribute_derivative_couplings.yml

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

You are free to choose your own HDF5 file name but for this tutorial we will use *chunk_01.hdf5* as an example. 

- Merge the different chunk into a single file using the *mergeHDF5.py* script:

  ``mergeHDF5.py -i chunk_0.hdf5 chunk_1.hdf5 -o chunk_01.hdf5``

Follow -i with the names of different chunks you want to merge and follow -o the name of the merged HDF5 file.  

- Remove the couplings from the chunk_01.hdf5 using the *removeHDF5folders.py* script. To run the script, use: 

  ``removeHDF5folders.py -hdf5 chunk_01.hdf5``

Using the script in this manner will only allow the couplings to be removed. 

.. Note::
   If required, you can remove all overlaps by by adding -o at the end of the previous command:

  ``removeHDF5folders.py -hdf5 chunk_01.hdf5 –o``


- Create a new subfolder in your original working directory and copy the *input.yml* file that was created for chunk 0 (when running the *distribute_jobs.py* script) to this folder. 

- Edit the *input.yml* file to include the path to the merged .hdf5, the full MD trajectory, and a new scratch path for the merged hdf5 calculations.

- Relaunch the calculation.

Once the remaining overlaps and the couplings have been calculated successfully, the hdf5 files and hamiltonians will be written to both the working directory as well as the scratch folder in a format suitable for PYXAID to run the non-adiabatic excited state molecular dynamics. If requested, also the overlap integrals can be found in the working directory.

.. note::
   There are several way to declare the parameters of the unit cell, you can passed to the cell_parameters
   variable either a number, a list or a list or list. A single number represent a cubic box, while a list
   represent a parallelepiped and finally a list of list contains the ABC vectors describing the unit cell.
   Alternatively, you can pass the angles of the cell using the cell_angles variable.

Restarting a Job
----------------

Both the *molecular orbitals* and the *derivative couplings* for a given molecular dynamic trajectory are stored in a HDF5_. The library check wether the *MO* orbitals or the coupling under consideration are already present in the HDF5_ file, otherwise compute it. Therefore  if the workflow computation fails due to a recoverable issue like:

  * Cancelation due to time limit.
  * Manual suspension or cancelation for another reasons.

Then, in order to restart the job you need to perform the following actions:

  * **Do Not remove** the file called ``cache.db`` from the current work  directory.


Reporting a bug or requesting a feature
---------------------------------------
To report an issue or request a new feature you can use the github issues_ tracker.

.. _HDF5: http://www.h5py.org/
.. _issues: https://github.com/SCM-NV/nano-qmflows/issues
.. _QMflows: https://github.com/SCM-NV/qmflows
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _YAML: https://pyyaml.org/wiki/PyYAML


