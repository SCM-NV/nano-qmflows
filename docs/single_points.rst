Single points calculation
=========================

The single_points workflow performs single point calculations and can be used, for example, to compute the eigenvalues and energies of triplet orbitals on a singlet geometry or viceversa.

Preparing the input
--------------------

Basic Example
^^^^^^^^^^^^^

We start with the very basic example of an input file to perform a single point calculation on the relaxed geometry of a Cd33Se33 system.

.. code-block:: yaml

    workflow:
      single_points

    project_name: Cd33Se33
    active_space: [50, 50]
    path_hdf5: "Cd33Se33.hdf5"
    path_traj_xyz: "Cd33Se33.xyz"
    scratch_path: "/tmp/singlepoints_basic"

    cp2k_general_settings:
      basis:  "DZVP-MOLOPT-SR-GTH"
      potential: "GTH-PBE"
      cell_parameters: 28.0
      periodic: none
      executable: cp2k.popt

      cp2k_settings_main:
        specific:
          template: pbe_main
          cp2k:
            force_eval:
              dft:
                scf:
                  eps_scf: 1e-6

      cp2k_settings_guess:
        specific:
          template:
            pbe_guess

In your working directory, create an *input_test_single_points_basic.yml* file and copy the previous input inside it, by respecting the indentation.
Also copy locally the file containing the coordinates of the Cd33Se33 system in an xyz format, Cd33Se33.xyz_.

Your *input_test_single_points_basic.yml* now contains all settings to perform the calculations and needs to be edited according to your system and preferences. 
Pay attention to the following parameters that are common to all input files: **workflow**, **project_name**, **active_space**, **path_hdf5**, **path_traj_xyz**, **scratch_path**.

- **workflow**: The workflow you need for your calculations, in this case single_points for a single point calculation.
- **project_name**: Project name for the calculations. You can choose what you wish.
- **active_space**: Range of (occupied, virtual) molecular orbitals to be computed. For example, if 50 occupied and 100 virtual should be considered in your calculations, the active space should be set to [50, 100].
- **path_hdf5**: Path where the hdf5 should be created / can be found. The hdf5 is the format used to store the molecular orbitals and other information.
- **path_traj_xyz**: Path to the pre-optimized geometry of your system. It should be provided in xyz format.
- **scratch_path**: A scratch path is required to perform the calculations. For large systems, the .hdf5 files can become quite large (hundredths of GBs) and calculations are instead performed in the scratch workspace. The final results will also be stored here.

You can find the complete list of all the general options (common to all workflows) in this dictionary_.

In the cp2k_general_settings, you can customize the settings used to generate the cp2k input (see available options in schema_cp2k_general_settings_).
Here you can specify the level of theory you want to use in your cp2k calculation (basis set and potential) as well as the main characteristics of your system (cell parameters and angles, periodicity, charge, multiplicity, …). 

Note that the (fast) SCF routine in cp2k is based on the Orbital Transformation (OT) method, which works on the occupied orbital subspace. To obtain the full spectrum of molecular orbitals, one should perform a full diagonalization of the Fock matrix. For this reason, to obtain and store both occupied and unoccupied orbitals, defined using the active_space keyword, we have to follow a 2-step procedure: in the first step, which in the yaml input we define as cp2k_settings_guess, we perform a single point calculation using the fast OT approach; then in the second step, defined as cp2k_settings_main, we use the converged orbitals in the first step to start a full diagonalization calculation using the DIIS procedure.

In the cp2k_settings_guess and cp2k_settings_main subsections you can provide more detailed information about the cp2k input settings to be used to compute the guess wavefunction and to perform the main calculation respectively.
You can also overwrite some parameters of the available templates_ according to your own requirements. In this example, the convergence criterion for the scf has been tightened to 1e-6 (the dafault value in the pbe_main template is 5e-4).
You can use the cp2k manual_ to further personalize your input requirements.

.. _Cd33Se33.xyz: https://github.com/SCM-NV/nano-qmflows/blob/master/test/test_files/Cd33Se33.xyz
.. _dictionary: https://github.com/SCM-NV/nano-qmflows/blob/e176ade9783677962d5146d8e6bc5dd6bb4f9102/nanoqm/workflows/schemas.py#L116
.. _schema_cp2k_general_settings: https://github.com/SCM-NV/nano-qmflows/blob/e176ade9783677962d5146d8e6bc5dd6bb4f9102/nanoqm/workflows/schemas.py#L55
.. _templates: https://github.com/SCM-NV/nano-qmflows/blob/master/nanoqm/workflows/templates.py
.. _manual: https://manual.cp2k.org/


Advanced Example
^^^^^^^^^^^^^^^^

We are now ready to move to a more advanced example of input file: here we want to compute the orbitals' energies and eigenvalues for each point of a pre-computed MD trajectory in which the cell parameters are allowed to change. For our Cd33Se33 system, the input will look like that:

.. code-block:: yaml

    workflow:
      single_points

    project_name: Cd33Se33
    active_space: [50, 50]
    dt: 1
    path_hdf5: "Cd33Se33.hdf5"
    path_traj_xyz: "Cd33Se33_fivePoints.xyz"
    scratch_path: "/tmp/singlepoints_advanced"

    cp2k_general_settings:
      basis:  "DZVP-MOLOPT-SR-GTH"
      potential: "GTH-PBE"
      cell_parameters: 28.0
      file_cell_parameters: "file_distribute_cell_parameters.txt"
      periodic: none
      executable: cp2k.popt

      cp2k_settings_main:
        specific:
          template: pbe_main

      cp2k_settings_guess:
        specific:
          template:
            pbe_guess
            

In your working directory, create an *input_test_single_points_advanced.yml* file and copy the previous input inside it (remember to respect the indentation). 
Also copy locally the small pre-computed MD trajectory of our Cd33Se33 system, Cd33Se33_fivePoints.xyz_ along with the text file containing the cell parameters at each point of the trajectory, file_distribute_cell_parameters.txt_.

In the input file, pay particular attention to the following parameters that have been added/modified with respect to the previous example:

- **dt**: The size of the timestep used in your MD simulations (in fs).
- **path_traj_xyz**: Path to the pre-computed MD trajectory. It should be provided in xyz format.

In the cp2k_general_settings, the path to the cell parameters file is now provided using **file_cell_parameters**.

.. _Cd33Se33_fivePoints.xyz: https://github.com/SCM-NV/nano-qmflows/blob/master/test/test_files/Cd33Se33_fivePoints.xyz
.. _file_distribute_cell_parameters.txt: https://github.com/SCM-NV/nano-qmflows/blob/master/test/test_files/file_distribute_cell_parameters.txt

Setting up the calculation 
---------------------------

Once all settings of your yml input have been customized, you are ready to launch your single point calculation.

- First, activate the conda environment with QMFlows:

  ``conda activate qmflows``
  
- Then, load the your version of the cp2k module, for example:

  ``module load CP2K/7.1.0``
  
- Finally, use the command run_workflow.py to submit your calculation:

  ``run_workflow.py -i input_test_single_points_basic.yml``
  
  or 

  ``run_workflow.py -i input_test_single_points_advanced.yml``
  
  for the advanced example.
