====================
Tutorial on QMFlows-NAMD calculations
====================

An MD trajectory is required to run these calculations. QMFlows_ and QMFlows-NAMD_ also need to be fully installed.

.. _QMFlows: https://github.com/SCM-NV/qmflows
.. _QMFlows-NAMD: https://github.com/SCM-NV/qmflows-namd

Preparing the input
--------------------

To perform the NAMD calculations, you will need the *distribute_jobs.py* file. You can find this file in the folder *escience/qmflows-namd/scripts/distribution/*. Copy this file to a folder where you want start the QMFlows calculations. 

The *distribute_jobs.py* file contains all settings to perform the calculations and needs to be edited according to your system and preferences. Pay attention to the following parameters: *algorithm, scratch, project_name, basisCP2K, potCP2K, cell_parameters, cell_angles, range_orbitals, nHOMO, coupling_range, path_to_trajectory, blocks, dt, slurm configurations*. 

- **algorithm**: Algorithm to calculate derivative couplings can be set to ‘levine’ or ‘3points’.
- **scratch**: A scratch path is required to perform the calculations. For large systems, the .hdf5 files can become quite large (in GBs) and calculations are instead performed in the scratch workspace. The final results will also be stored here. 
- **project_name**: Project name for the calculations. 
- **basisCP2K**, **potCP2K**: Path to the CP2K basis set and potentials files. 
- **cell_parameters**, **cell_angles**: Cell parameters and cell angles for the CP2K single point calculations. These should be the same values as in your MD simulation. 
- **mo_index_range**: Absolute range of orbitals for which the overlaps/couplings will be calculated. Example: the HOMO lies at 2725 and our range of interest consists of 100 HOMOs and 100 LUMOs. Our range_orbitals would be set to ‘2626 2825’.
It is the same keyword than CP2K mo_index_range_.
- **nHOMO**: The index of the HOMO within the orbital range has to be provided. In the above example, the index of the HOMO is 100.
- **path_to_trajectory**: Location of the .xyz file of the trajectory. 
- **blocks**: Number of blocks (chunks) is related to how the trajectory is split up. As our typical trajectories are quite large (+- 2000 structures), it is convenient to split the trajectory up into multiple chunks so that several calculations can be performed simultaneously. Generally around 4-5 blocks is sufficient, depending on the length of your trajectory and the size of your system. 
- **dt**: Time steps used in your MD trajectory. 
- **slurm configurations**: The slurm configuration needs to be customized according to the supercomputer used at your facility. You can also provide a custom name for your calculations.

In the *cp2k_input* section at the bottom of the *distribute_jobs.py* file, you can change the CP2K settings used for the wavefunction guess and main CP2K job. Default basisset and functional are set to DZVP-MOLOPT-SR-GTH and GTH-PBE.

Setting up the calculation 
---------------------------

Once all settings in *distribute_jobs.py* have been customized, you will need to create the different chunks. 
  
- First, activate QMFlows:

  ``conda activate qmflows``  

- Use python to run the script, using the command:

  ``python distribute_jobs.py``

A number of new folders are created. In each folder you will find a launch.sh file, a chunk_xyz file and a script_remote_function.py file. In the script_remote_function.py file, you will find all your settings. 

 Note:
 As a default, QMFlows tracks the states at each time step of the trajectory. To turn off the tracking of states, add ``tracking=False,`` in the *generate_pyxaid_hamiltonians* section of the *script_remote_function.py* file.

 Another option would be to add ``write_overlaps=True,`` and ``overlaps_deph=True,`` to the *generate_pyxaid_hamiltonians* section to calculate the overlaps according to … (reference).

- When you are done editing the files, submit each of your jobs:

  ``sbatch launch.sh``

You can keep track of the calculations by going to your scratch path. The CP2K calculations of each chunk can be found in their respective batch folders. 

The overlaps and couplings between each state will be calculated once the single point calculations are finished. The progress can be tracked with the .log file in your home folders. The calculated couplings are meaningless at this point and need to be removed and recalculated, more on that later.  

Merging the chunks and recalculating the couplings 
---------------------------------------------------

Once the overlaps and couplings are all calculated, you need to merge the different chunks into a single chunk, as the overlaps between the different chunks still need to be calculated. For this you will use the *mergeHDF5.py* command that you will have if you have installed QMFlows correctly. 

- Create a new .hdf5 file using the touch command to create a new .hdf5 file:

  ``touch chunk_abcde.hdf5``

You are free to choose your own HDF5 file name but for this tutorial we will use *chunk_abcde.hdf5* as an example. 

- Merge the different chunk into a single file using the *mergeHDF5.py* script:

  ``mergeHDF5.py -i chunk_a.hdf5 chunk_b.hdf5 chunk_c.hdf5 chunk_d.hdf5 chunk_e.hdf5 -o chunk_abcde.hdf5``

Follow -i with the names of different chunks you want to merge and follow -o the name of the merged HDF5 file.  

- Remove the couplings from the chunk_abcde.hdf5 using the *removeHDF5folders.py* script. To run the script, use: 

  ``removeHDF5folders.py -pn PROJECTNAME -HDF5 chunk_abcde.hdf5``

Replace PROJECTNAME with your project name. 

Using the script in this manner will only allow the couplings to be removed. 

 Note: If required, you can remove all overlaps by by adding -o at the end of the previous command:

 ``removeHDF5folders.py -pn PROJECTNAME -hdf5 chunk_abcde.hdf5 –o``

- Create a new subfolder in your original QMFlows folder and copy the *script_remote_function.py* file that was created for chunk a (when running the *distribute_jobs.py* script) to this folder. 

- Edit the *script_remote_function.py* file to include the path to the merged .hdf5, the full MD trajectory, and a new scratch path for the merged hdf5 calculations.

- Relaunch the calculation.

Once the remaining overlaps and the couplings have been calculated successfully, results will be written to both the working folder as well as the scratch folder. The overlaps will be written to the same folder as your *script_remote_function.py*. The Hamiltonians will be written to the scratch folder belonging to the merged HDF5.

.. _mo_index_range: https://manual.cp2k.org/cp2k-6_1-branch/CP2K_INPUT/FORCE_EVAL/DFT/PRINT/MO.html#list_MO_INDEX_RANGE
