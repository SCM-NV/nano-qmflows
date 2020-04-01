Distribute Absorption Spectrum 
==============================

This workflow computes the absorption spectra for a given molecular system and returns a set of files in TXT format. The principle of distribution workflow is dividing the work in multiple, separated, instances (chunks), in order to be able to split time-consuming jobs into smaller, quicker ones.
The input is described in YAML format as showed in the following example:

.. code-block:: yaml 

 workflow:
   distribute_absorption_spectrum
 project_name:
   DMSO_PbBr4
 xc_dft:
   pbe
 tddft:
   stda
 active_space: [20, 20]
 stride:
   10
 path_hdf5:
   "/scratch-shared/frazac/tutorial/guanine.hdf5"
 path_traj_xyz:
   "/home/frazac/IVAN/Guanine/MD/qmworks-cp2k-pos-1.xyz"
 scratch_path:
   "/scratch-shared/frazac/guanine/"
 workdir: "."
 blocks: 5

 job_scheduler:
   scheduler: SLURM
   nodes: 1
   tasks: 24
   wall_time: "1:00:00"
   queue_name: "short"
   load_modules: "source activate qmflows\nmodule load eb\nmodule load CP2K/5.1-foss-2017b"

 cp2k_general_settings:
   basis:  "DZVP-MOLOPT-SR-GTH"
   potential: "GTH-PBE"
   path_basis: "/home/frazac/cp2k_basis"
   periodic: "xyz"
   charge: 0
   cell_parameters: 25.0
   cell_angles: [90.0,90.0,90.0]

   cp2k_settings_main:
     specific:
       template: pbe_main

   cp2k_settings_guess:
      specific:
       template: pbe_guess

The input *template_distribute_absorption_spectrum.yml* file contains all settings to perform the calculations and needs to be edited according to your system and preferences. Pay attention to the following parameters, which are specific for this workflow:

- **stride**: this parameter controls the accuracy of sampling of geometries contained in the MD trajectory of reference. For example, a value of stride: 10 indicates that the spectrum analysis will be performed on 1 out of 10 points in the reference trajectory. Two important things have to be pointed out:

  #. The workflow will perform SCF calculations for each point in the trajectory; only afterwards it will sample the number of structures on which the spectrum analysis will be performed

  #. Down-sampling issues might arise from the number of points that are actually printed during the MD calculations. Some programs, indeed, offer the possibility to print (in the output file) only one point out of ten (or more) calculated. In this case, applying a stride: 10 would in practice mean that you are sampling 1 point out of 100 points in the trajectory.

- **blocks**: this parameter indicates into how many blocks has the job to be split. This will generate as many chunks’ folders in your working directory, all of each containing th

Note: TRIPLETs

Reporting a bug or requesting a feature
---------------------------------------
To report an issue or request a new feature you can use the github issues_ tracker.

.. _HDF5: http://www.h5py.org/
.. _issues: https://github.com/SCM-NV/nano-qmflows/issues
.. _QMflows: https://github.com/SCM-NV/qmflows
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _YAML: https://pyyaml.org/wiki/PyYAML


