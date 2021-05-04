Crystal Orbital Overlap Population (COOP) calculation
=====================================================

The workflow coop_calculation allows to compute the crystal orbital overlap population between two selected elements.

Preparing the input
-------------------

The following is an example of input file to perform the COOP calculation between Cd and Se for the Cd33Se33 system.

.. code-block:: yaml

    workflow:
      coop_calculation

    project_name: Cd33Se33
    active_space: [50, 50]
    path_hdf5: "Cd33Se33.hdf5"
    path_traj_xyz: "Cd33Se33.xyz"
    scratch_path: "/tmp/COOP"

    coop_elements: ["Cd", "Se"]

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


In your working directory, copy the previous input into an *input_test_coop.yml* file. 
Also copy locally the file containing the coordinates of the relaxed Cd33Se33 system, Cd33Se33.xyz_.

Your *input_test_coop.yml* input file now contains all settings to perform the coop calculations and needs to be edited according to your system and preferences.
Please note that this input is very similar to the basic example of single point calculation provided in a previous tutorial_ (please refer to it for a more extensive description of the above options)
except for the following options: **workflow**, **coop_elements**.

- **workflow**: The workflow you need for your calculations, in this case set to coop_calculation is this case.
- **coop_elements**: List of the two elements to calculate the COOP for, here Cd and Se.

In the cp2k_general_settings, you can customize the settings used to generate the cp2k input. To help you creating your custom input requirements, please consult the cp2k manual_ and the templates_ available in nano-qmflows.

.. _Cd33Se33.xyz: https://github.com/SCM-NV/nano-qmflows/blob/master/test/test_files/Cd33Se33.xyz
.. _tutorial: https://qmflows-namd.readthedocs.io/en/latest/single_points.html
.. _manual: https://manual.cp2k.org/
.. _templates: https://github.com/SCM-NV/nano-qmflows/blob/master/nanoqm/workflows/templates.py

Setting up the calculation 
---------------------------

Once all settings of your yml input have been customized, can to launch your coop calculation.

- First, activate the conda environment with QMFlows:

  ``conda activate qmflows``
  
- Then, load the module with your version of cp2k, for example:

  ``module load CP2K/7.1.0``
  
- Finally, use the command run_workflow.py to submit your calculation.

  ``run_workflow.py -i input_test_coop.yml``

Results 
-------

Once your calculation has finished successfully, you will find a *COOP.txt* file in your working directory.
The two columns of this file contain, respectively, the orbitals’ energies and the corresponding COOP values for the selected atoms pair.
