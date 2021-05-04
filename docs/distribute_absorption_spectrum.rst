Distribute Absorption Spectrum 
==============================

This workflow computes the absorption spectra for a given molecular system and returns a set of files in TXT format. The principle of distribution workflow is dividing the work in multiple, separated, instances (chunks), in order to be able to split time-consuming jobs into smaller, quicker ones.

In this tutorial, we want to compute the excited states at **each point** of a pre-computed MD trajectory for the guanine system. Please note that this trajectory has already been used in the advanced example of the absorption_spectrum tutorial_, where the spectrum analysis was performed on 1 out of 4 points only. Here we take advantage of the distribution workflow to increase by four times the accuracy of sampling with no substantial variation of the computational cost in terms of time by dividing the job in five chunks (each taking charge of 4 points out of 20).

Preparing the input
--------------------

The input is described in YAML format as showed in the following example:

.. code-block:: yaml 

    workflow:
      distribute_absorption_spectrum
    
    project_name: guanine_distribution
    active_space: [20, 20]
    dt: 1
    path_hdf5: "guanine.hdf5"
    path_traj_xyz: "guanine_twentyPoints.xyz"
    scratch_path: "/tmp/distribute_absorption_spectrum"
    calculate_guesses: "first"
    
    xc_dft: pbe
    tddft: stda
    stride: 1
    
    blocks: 5
    workdir: "."
    
    job_scheduler:
      free_format: "
      #! /bin/bash\n
      #SBATCH --job-name=guanine_distribution\n
      #SBATCH -N 1\n
      #SBATCH -t 1:00:00\n
      #SBATCH -p short\n
      source activate qmflows\n
      module load CP2K/7.1.0\n\n"
    
    cp2k_general_settings:
      basis:  "DZVP-MOLOPT-SR-GTH"
      potential: "GTH-PBE"
      cell_parameters: 25.0
      periodic: none
      executable: cp2k.popt
    
      cp2k_settings_main:
        specific:
          template: pbe_main
    
      cp2k_settings_guess:
        specific:
          template: pbe_guess


In your working directory, create an *input_test_distribute_absorption_spectrum.yml* file and copy the previous input inside it (remember to respect the indentation). 
Also copy locally the small pre-computed MD trajectory of the guanine system, guanine_twentyPoints.xyz.

In the input file, pay particular attention to the following parameters that have been added/modified with respect to the previous tutorial_ (advanced example):

- **stride**: Controls the accuracy of sampling of geometries contained in the MD trajectory of reference. Here, a value of stride: 1 indicates that the spectrum analysis will be performed on each point in the reference trajectory. Two important things have to be pointed out:

  #. The workflow will perform SCF calculations for each point in the trajectory; only afterwards it will sample the number of structures on which the spectrum analysis will be performed

  #. Down-sampling issues might arise from the number of points that are actually printed during the MD calculations. Some programs, indeed, offer the possibility to print (in the output file) only one point out of ten (or more) calculated. For example, applying a stride: 10 would in practice mean that you are sampling 1 point out of 100 points in the trajectory.

- **blocks**: Indicates into how many blocks has the job to be split. This will generate as many chunks’ folders in your working directory.

- **workdir**: Path to the chunks' folders.

The **job_scheduler** can also be found below these parameters. Customize these settings according to the system and environment you are using to perform the calculations.

.. _tutorial: https://qmflows-namd.readthedocs.io/en/latest/absorption_spectrum.html
.. _tutorial: https://qmflows-namd.readthedocs.io/en/latest/absorption_spectrum.html

Setting up the calculation 
---------------------------

Once all settings in *input_test_distribute_absorption_spectrum.yml* have been customized, you will need to create the different chunks. 
  
- First, activate QMFlows:

  ``conda activate qmflows``  

- Use the command *distribute_jobs.py* to create the different chunks:

  ``distribute_jobs.py -i input_test_distribute_absorption_spectrum.yml``

A number of new folders are created. In each folder you will find a submission file, launch.sh, a sub-trajectory file (containing the points assigned to that chunk), chunk_xyz, and an input.yml file. In the input.yml file, you will find all your settings. Check for any possible manual errors.

- If you are satisfied with the inputs, submit each of your jobs for calculation.

You can keep track of the calculations by going to your scratch path. The location where all points of the chunks are calculated is your assigned scratch path plus project name plus a number.

Results 
-------

Once the calculations are completed, you will find multiple *output_n_stda.txt* files in your scratch directories (with *n* being the index of the geometry at which the spectrum analysis has been performed). The first two lines of the file *output_0_stda.txt*, found in /tmp/distribute_absorption_spectrum/scratch_chunk_0/ are reported below.

::

    # state    energy       f      t_dip_x    t_dip_y    t_dip_y    weight   from   energy  to     energy     delta_E
        1      4.566    0.03832   -0.51792   -0.25870    0.08573    0.50158  20     -5.175  21     -1.261      3.914

For each excited state (line), the first six columns contain, from left to right:

- *# state*: Assigned index, in ascending order of energy. Here, the lowest excitation is reported and corresponds to # state 1.
- *energy*: Transition energy, in eV.
- *f*: Oscillator strength, dimensionless.
- *t_dip_x*, *t_dip_y*, *t_dip_z*: Transition dipole moment components along x, y and z.

The next six columns report some useful information about the dominant single orbital transition for the excited state under examination:

- *weight*: Weight in the overall transition. Always 1.0000 in the Single Orbital approximation.
- *from*: Index of the initial occupied orbital in the active space.
- *energy*: Energy of the initial occupied orbital.
- *to*: Index of the final virtual orbital in the active space.
- *energy*: Energy of the final virtual orbital.
- *delta_E*:Energy of the dominant single orbital transition. Corresponds to the excited state energy in the Single Orbital approximation.

Copy all the output files to your working directory and plot the absorption spectrum (averaged over all sampled structures) using the script convolution.py_:

  ``convolution.py -nm True``
  
To plot the absorption spectrum of a specific sample, for example our point 0, use the -n option.

  ``convolution.py -n 0 -nm True``

.. _convolution.py: https://github.com/SCM-NV/nano-qmflows/blob/master/scripts/qmflows/convolution.py

Reporting a bug or requesting a feature
---------------------------------------
To report an issue or request a new feature you can use the github issues_ tracker.

.. _HDF5: http://www.h5py.org/
.. _issues: https://github.com/SCM-NV/nano-qmflows/issues
.. _QMflows: https://github.com/SCM-NV/qmflows
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _YAML: https://pyyaml.org/wiki/PyYAML


