Introduction to the Tutorials
=============================

The *nano-qmflows* packages offers the following set of workflows to compute different properties:
 * single_points
 * coop_calculation
 * ipr_calculation
 * derivative_coupling
 * absorption_spectrum
 * distribute_absorption_spectrum

Known Issues
------------

Distribution of the workflow over multiple nodes
################################################

`CP2K` can uses multiple nodes to perform the computation of the molecular orbitals using the **MPI** protocol. Unfortunately, the `MPI` implementation for the computation of the *derivative coupling matrix* is experimental and unestable. The practical consequences of the aforemention issues, is that **the calculation of the coupling matrices are carried out in only 1 computational node**. It means that if you want ask for more than 1 node to compute the molecular orbitals with `CP2K`, once the workflow starts to compute the *derivative couplings* only 1 node will be used at a time and the rest will remain idle wating computational resources.


Reporting a bug or requesting a feature
---------------------------------------
To report an issue or request a new feature you can use the github issues_ tracker.

.. _HDF5: http://www.h5py.org/
.. _issues: https://github.com/SCM-NV/nano-qmflows/issues
.. _QMflows: https://github.com/SCM-NV/qmflows
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _YAML: https://pyyaml.org/wiki/PyYAML


