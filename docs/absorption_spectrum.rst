Absorption Spectrum
===================

This other workflow computes the excited states energies, transition dipole moments and oscillator strength using the STDDFT approach.

Preparing the input
--------------------

Basic Example
^^^^^^^^^^^^^

Below is a basic example of input file for the computation of the first 400 (20*20, as setted in the active_space) lowest lying excited states of a guanine molecule at the sTDA level of approximation.

.. code-block:: yaml

    workflow:
      absorption_spectrum

    project_name: guanine
    active_space: [20, 20]
    compute_orbitals: True
    path_hdf5: "guanine.hdf5"
    path_traj_xyz: "guanine.xyz"
    scratch_path: "/tmp/absorption_spectrum_basic"

    xc_dft: pbe
    tddft: stda

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

In your working directory, create an *input_test_absorption_spectrum_basic.yml* file and copy the previous input inside it, by paying attention to preserve the correct indentation.
Also copy locally the file containing the coordinates of the relaxed geometry of the guanine in an xyz format, guanine.xyz.

At this point, your *input_test_absorption_spectrum_basic.yml* contains all the settings to perform the excited states calculation and needs to be edited according to your system and preferences. First, let’s recall some parameters that are common to all input files: **workflow**, **project_name**, **active_space**, **path_hdf5**, **path_traj_xyz**, **scratch_path**.

- **workflow**: The workflow you need for your calculations, in this case absorption_spectrum to compute excited states properties.
- **project_name**: Project name for the calculations. You can choose what you wish.
- **active_space**: Range of (doubly occupied, virtual) molecular orbitals to be computed. For example, if 50 occupied and 100 virtual should be considered in your calculations, the active space should be set to [50, 100]. The range will automatically be appended with additional singly occupied MOs based on the systems (user-specified) multiplicity.
- **compute_orbitals**: Specify if the energy and eigenvalues of the selected orbitals are to be computed. The default is set to True so we will not consider it in the advanced examples.
- **path_hdf5**: Path where the hdf5 should be created / can be found. The hdf5 is the format used to store the molecular orbitals and other information.
- **path_traj_xyz**: Path to the pre-optimized geometry of your system. It should be provided in xyz format.
- **scratch_path**: A scratch path is required to perform the calculations. For large systems, the .hdf5 files can become quite large (hundredths of GBs) and calculations are instead performed in the scratch workspace. The final results will also be stored here.

You can find the complete list of these general options in this dictionary_.

Also pay particular attention to the following parameters, specific to the absorption_spectrum workflow:

- **xc_dft**: Type of exchange-correlation functional used in your DFT calculations.
- **tddft**:  Type of approximation used in the excited states calculations. The Single Orbital (sing_orb), sTDA (stda) and sTDDFT (stddft) approximations are available.

In the cp2k_general_settings, you can customize the settings used to generate the cp2k input of the initial single point calculations (from which Molecular Orbital energies and coefficients are retrieved). For more details about this section please refer to the available tutorial_ on single point calculations. To further personalize the input requirements, also consult the cp2k manual_ and the templates_ available in nano-qmflows.

.. _dictionary: https://github.com/SCM-NV/nano-qmflows/blob/e176ade9783677962d5146d8e6bc5dd6bb4f9102/nanoqm/workflows/schemas.py#L116
.. _schema_cp2k_general_settings: https://github.com/SCM-NV/nano-qmflows/blob/e176ade9783677962d5146d8e6bc5dd6bb4f9102/nanoqm/workflows/schemas.py#L55
.. _templates: https://github.com/SCM-NV/nano-qmflows/blob/master/nanoqm/workflows/templates.py
.. _manual: https://manual.cp2k.org/
.. _tutorial: https://github.com/SCM-NV/nano-qmflows/blob/master/docs/single_points.rst


Advanced Example
^^^^^^^^^^^^^^^^

We are now ready to move to a more advanced example in which we want to compute the excited states of our guanine molecule starting from a pre-computed MD trajectory rather than a single geometry. The input file will look like that:

.. code-block:: yaml

    workflow:
      absorption_spectrum

    project_name: guanine
    active_space: [20, 20]
    dt: 1
    path_hdf5: "guanine.hdf5"
    path_traj_xyz: "guanine_twentyPoints.xyz"
    scratch_path: "/tmp/absorption_spectrum_advanced1"
    calculate_guesses: "first"

    xc_dft: pbe
    tddft: stda
    stride: 4

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

In your working directory, create an *input_test_absorption_spectrum_advanced.yml* file and copy the previous input inside it (remember to respect the indentation).
Also copy locally the small pre-computed MD trajectory of the guanine system, guanine_twentyPoints.xyz.

In the input file, pay particular attention to the following parameters that have been added/modified with respect to the previous example:

- **dt**: The size of the timestep used in your MD simulations (in fs).
- **path_traj_xyz**: Path to the pre-computed MD trajectory. It should be provided in xyz format.
- **calculate_guesses**: Specify whether to calculate the guess wave function only in the first point of the trajectory ("first") or in all ("all). Here, we keep the default value, first.
- **stride**: Controls the accuracy of sampling of geometries contained in the MD trajectory of reference. For example, our value of stride: 4 indicates that the spectrum analysis will be performed on 1 out of 4 points in the reference trajectory. Two important things have to be pointed out:

  #. The workflow will perform SCF calculations for each point in the trajectory (twenty points in our example); only afterwards it will sample the number of structures on which the spectrum analysis will be performed (here six structures corresponding to points 0, 4, 8, 12, 16, 20).

  #. Down-sampling issues might arise from the number of points that are actually printed during the MD calculations. Some programs, indeed, offer the possibility to print (in the output file) only one point out of ten (or more) calculated. In this case, applying a stride: 4 would in practice mean that you are sampling 1 point out of 40 points in the trajectory.

Setting up the calculation
---------------------------

Once all settings of your yml input have been customized, you are ready to launch your single point calculation.

- First, activate the conda environment with QMFlows:

  ``conda activate qmflows``

- Then, load the module with your version of cp2k, for example:

  ``module load CP2K/7.1.0``

- Finally, use the command run_workflow.py to submit your calculation:

  ``run_workflow.py -i input_test_absorption_spectrum_basic.yml``

for the basic example.

Results
-------

Once your calculation has finished successfully, you will find one (or more) *output_n_stda.txt* file(s) in your scratch directory (with *n* being the index of the geometry at which the spectrum analysis has been performed). The first two lines of the file *output_0_stda.txt* generated in our basic example are reported below.

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

Copy the output file(s) to your working directory and plot the absorption spectrum using the script convolution.py_:

  ``convolution.py -nm True``

In case of multiple output files, the returned absorption spectrum is an average over all sampled strucutures, unless you define the index of a specific sample using the -n option.

.. _convolution.py: https://github.com/SCM-NV/nano-qmflows/blob/master/scripts/qmflows/convolution.py

Reporting a bug or requesting a feature
---------------------------------------
To report an issue or request a new feature you can use the github issues_ tracker.

.. _HDF5: http://www.h5py.org/
.. _issues: https://github.com/SCM-NV/nano-qmflows/issues
.. _QMflows: https://github.com/SCM-NV/qmflows
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _YAML: https://pyyaml.org/wiki/PyYAML
