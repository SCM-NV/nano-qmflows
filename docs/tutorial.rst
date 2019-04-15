
Tutorial
=========

The *qmflows-namd* packages offers a set of workflows to compute different properties like:

 * derivative_Coupling
 * absorption_spectrum

   
Derivative coupling calculation
-------------------------------

This workflow computes the derivative coupling matrix for a given molecular system and returns
a set of files in PYXAID_ format. The input is described in YAML_ format as showed in the
following example:

.. literalinclude:: ../test/test_files/input_fast_test_derivative_couplings.yml
    :linenos:

The `workflow` keyword in lines 1-2 described the name of the workflow to run, while in lines 3-4 the index
of the **HOMO** orbital is provided (starting from 1). Subsequently the `coupling_range` keyword states
the number of orbitals to compute the couple, where the first number is the index of the starting orbital and the
second number the number of orbitals to include *after the first number*. In the previous example, the Coupling
for orbitals 50 to 79 will be computed.

the keyword `path_hdf5` point to the file where all the orbitals and couplings will be stored. Moreover, the
`path_traj_xyz` variable is the name of the file containing the molecular dynamic trajectory in *xyz* format.

From Line 22 to the end the input for CP2K is provided.Using the aforemention Settings QMflows_ automatically create the CP2K input. You do not need to add the & or &END symbols, QMFlows adds them automatically for you.

CP2K requires the explicit declaration of the basis set together with the name of the potential used for each one of the atoms. In the previous example the basis for the carbon is DZVP-MOLOPT-SR-GTH, while the potential is GTH-PBE.

.. note::
   There are several way to declare the parameters of the unit cell, you can passed to the cell_parameters
   variable either a number, a list or a list or list. A single number represent a cubic box, while a list
   represent a parallelepiped and finally a list of list contains the ABC vectors describing the unit cell.
   Alternatively, you can pass the angles of the cell using the cell_angles variable.

Derivative coupling calculation
-------------------------------
This other workflow compute the oscillator strenghts for different snapshots in a MD trajectory.

.. literalinclude:: ../test/test_files/input_test_absorption_spectrum.yml
    :linenos:


Restarting a Job
----------------

Both the *molecular orbitals* and the *derivative couplings* for a given molecular dynamic trajectory are stored in a HDF5_. The library check wether the *MO* orbitals or the coupling under consideration are already present in the HDF5_ file, otherwise compute it. Therefore  if the workflow computation fails due to a recoverable issue like:

  * Cancelation due to time limit.
  * Manual suspension or cancelation for another reasons.

Then, in order to restart the job you need to perform the following actions:

  * **Do Not remove** the file called ``cache.db`` from the current work  directory.
  

Distribute Absorption Spectrum
------------------------------

This workflow computes the absorption spectra for a given molecular system and returns a set of files in TXT format. The principle of distribution workflow is dividing the work in multiple, separated, instances (chunks), in order to be able to split time-consuming jobs into smaller, quicker ones.
The input is described in YAML format as showed in the following example:

.. literalinclude:: ../test/test_files/guanine_distribution.yml
    :linenos:

The input file guanine_distribution.yml contains all settings to perform the perform the calculation of absorption spectra of Guanine over a trajectory described by multiple geometries, and needs to be edited according to your system and preferences. Pay attention to the following parameters, which are specific for this workflow:

 * **Stride:** this parameter controls the accuracy of sampling of geometries contained in the MD trajectory of reference. For example, a value of stride: 10 indicates that the spectrum analysis will be performed on 1 out of 10 points in the reference trajectory. 
 
Two important things have to be pointed out:
  1.	The workflow will perform SCF calculations for each point in the trajectory; only afterwards it will sample the number of structures on which the spectrum analysis will be performed
  2.	Down-sampling issues might arise from the number of points that are actually printed during the MD calculations. Some programs, indeed, offer the possibility to print (in the output file) only one point out of ten (or more) calculated. In this case, applying a - stride: 10 - would in practice mean that you are sampling 1 point out of 100 points in the trajectory.

 * **Blocks:** this parameter indicates into how many blocks has the job to be split. This will generate as many chunks’ folders in your working directory, all of each containing the specific input and launch files.
 

Known Issues
------------

Coupling distribution in multiple nodes
#########################################

`CP2K` can uses multiple nodes to perform the computation of the molecular orbitals using the **MPI** protocol. Unfortunately, the `MPI` implementation for the computation of the *derivative coupling matrix* is experimental and unestable. The practical consequences of the aforemention issues, is that **the calculation of the coupling matrices are carried out in only 1 computational node**. It means that if you want ask for more than 1 node to compute the molecular orbitals with `CP2K`, once the workflow starts to compute the *derivative couplings* only 1 node will be used at a time and the rest will remain idle wating computational resources. 


Reporting a bug or requesting a feature
---------------------------------------
To report an issue or request a new feature you can use the github issues_ tracker.

.. _HDF5: http://www.h5py.org/
.. _issues: https://github.com/SCM-NV/qmflows-namd/issues
.. _QMflows: https://github.com/SCM-NV/qmflows
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _YAML: https://pyyaml.org/wiki/PyYAML
