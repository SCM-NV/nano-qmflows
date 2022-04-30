Theory
==========

Nonadiabatic coupling matrix
-----------------------------

The current implementation of the nonadiabatic coupling is based on:
Plasser, F.; Granucci, G.; Pittner, j.; Barbatti, M.; Persico, M.;
Lischka. *Surface hopping dynamics using a locally diabatic formalism:
Charge transfer in the ethylene dimer cation and excited state dynamics
in the 2-pyridone dimer*. **J. Chem. Phys. 2012, 137, 22A514.**

The total time-dependent wave function :math:`\Psi(\mathbf{R}, t)` can be
expressed in terms of a linear combination of ``N`` adiabatic electronic
eigenstates :math:`\phi_{i}(\mathbf{R}(t))`,

.. math::
   \Psi(\mathbf{R}, t) = \sum^{N}_{i=1} c_i(t)\phi_{i}(\mathbf{R}(t)) \quad \mathbf(1)

The time-dependent coefficients are propagated according to

.. math::

   \frac{dc_j(t)}{dt} = -i\hbar^2 c_j(t) E_j(t) - \sum^{N}_{i=1}c_i(t)\sigma_{ji}(t) \quad \mathbf(2)

where :math:`E_j(t)` is the energy of the jth adiabatic state and :math:`\sigma_{ji}(t)` the nonadiabatic matrix, which elements are given by the expression

.. math::
  \sigma_{ji}(t) = \langle \phi_{j}(\mathbf{R}(t)) \mid \frac{\partial}{\partial t} \mid \phi_{i}(\mathbf{R}(t)) \rangle \quad \mathbf(3)

that can be approximate using three consecutive molecular geometries

.. math::
   \sigma_{ji}(t) \approx \frac{1}{4 \Delta t} (3\mathbf{S}{ji}(t) - 3\mathbf{S}{ij}(t) - \mathbf{S}{ji}(t-\Delta t) + \mathbf{S}{ij}(t-\Delta t)) \quad \mathbf(4)

where :math:`\mathbf{S}_{ji}(t)` is the overlap matrix between two consecutive time steps

.. math::
   \mathbf{S}{ij}(t) = \langle \phi{j}(\mathbf{R}(t-\Delta t)) \mid \phi_{i}(\mathbf{R}(t)) \rangle \quad \mathbf(5)

and the overlap matrix is calculated in terms of atomic orbitals

.. math::
   \mathbf{S}{ji}(t) = \sum{\mu} C^{*}{\mu i}(t) \sum{\nu} C_{\nu j}(t - \Delta t) \mathbf{S}_{\mu \nu}(t) \quad \mathbf(6)

Where :math:C_{\mu i} are the Molecular orbital coefficients and :math:`\mathbf{S}_{\mu \nu}` The atomic orbitals overlaps.

.. math::
   \mathbf{S}{\mu \nu}(\mathbf{R}(t), \mathbf{R}(t - \Delta t)) = \langle \chi{\mu}(\mathbf{R}(t)) \mid \chi_{\nu}(\mathbf{R}(t - \Delta t)\rangle \quad \mathbf(7)


Nonadiabatic coupling algorithm implementation
----------------------------------------------

The  figure belows shows schematically the workflow for calculating the Nonadiabatic
coupling matrices from a molecular dynamic trajectory. The uppermost node represent
a molecular dynamics
trajectory that is subsequently divided in its components andfor each geometry the molecular
orbitals are computed. These molecular orbitals are stored in a HDF5_.
binary file and subsequently calculations retrieve sets of three molecular orbitals that are
use to calculate the nonadiabatic coupling matrix using equations **4** to **7**.
These coupling matrices are them feed to the PYXAID_ package to carry out nonadiabatic molecular dynamics.

The Overlap between primitives are calculated using the Obara-Saika recursive scheme and has been implemented using the C++ libint2_ library for efficiency reasons.
The libint2_ library uses either OpenMP_ or C++ threads to distribute the integrals among the available CPUs.
Also, all the heavy numerical processing is carried out by the highly optimized functions in NumPy_.

 The **nonadiabaticCoupling** package relies on *QMWorks* to run the Quantum mechanical simulations using the [CP2K](https://www.cp2k.org/) package.  Also, the noodles_ is used
 to schedule expensive numerical computations that are required to calculate the nonadiabatic coupling matrix.


.. _OpenMP: https://www.openmp.org/
.. _libint2: https://github.com/evaleev/libint/wiki
.. _HDF5: http://www.h5py.org/
.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
.. _multiprocessing: https://docs.python.org/3.6/library/multiprocessing.html
.. _NumPy: http://www.numpy.org
.. _noodles: http://nlesc.github.io/noodles/
