# Change Log

# 0.10.4 (Unrelease)
## New
* Template to create B3LYP computations (#269)
* Allow to compute both alphas/betas derivative couplings simultaneusly (#275)
* Add nanoCAT dependency (#280)

## Changed
* Do not remove the CP2K log files by default
* Do not remove the point folder wher ethe CP2K orbitals are stored

## Fixed
* Unrestricted Hamiltitonians name (#286)
* Hamiltonian units (#290)
* Schema error (#292)


# 0.10.3 (09/10/2020)
## New
* Template to create B3LYP computations (#269)
* Add support for derivative couplings for system with more than one spin state (#275)

## Fixed
* Distribution error (#272)
* Molecular orbital error [in qmflows](https://github.com/SCM-NV/qmflows/pull/213) (#270)
* Multivalue settings issue (#260)
* CP2K executable (#264)

# 0.10.1
## New
* Keywords to print eigenvalues and eigenvectors (#248)

## Fixed
* SLURM free-format specification (#245)
* Touch HDF5 file if it doesn't exist (#246)
* Create a separate folder for each distributed chunk (#247)
* Error creating the scratch path and the HDF5 file (#255)

# 0.10.0
## Changed
* Rename package to **nano-qmflows**

# 0.9.0
## Added
* [Support for Mac](https://github.com/SCM-NV/nano-qmflows/issues/231)
* [Allow to specify the CP2K executable](https://github.com/SCM-NV/nano-qmflows/issues/226)
* [MyPy static checking](https://github.com/SCM-NV/nano-qmflows/issues/237)

## Changed
* [Use New QMFlows API](https://github.com/SCM-NV/nano-qmflows/issues/227)
* [Use Libint==2.6.0](https://github.com/SCM-NV/nano-qmflows/issues/234)
* [Allow the user to enter her own slurm script](https://github.com/SCM-NV/nano-qmflows/issues/225)

# 0.8.3

## Changed
* Add the global run_type  keyword in the templates

# 0.8.2 [31/01/20]

## Changed

* Replace `qmflows.utils` with [more-itertools](https://more-itertools.readthedocs.io/en/stable/index.html)

## Added
* [smiles_calculator](https://github.com/SCM-NV/nano-qmflows/blob/master/scripts/qmflows/smiles_calculator.py) script to compute molecular properties from smiles.

# 0.8.1 [17/10/19]

## Changed
* Use [f-strings](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)
* Clean [C++ interface](https://cgithub.com/SCM-NV/nano-qmflows/blob/master/libint/compute_integrals.cc) to [libint](https://github.com/evaleev/libint)

## Removed
* Unused code to compile the [C++ interface](https://cgithub.com/SCM-NV/nano-qmflows/blob/master/libint/compute_integrals.cc) to [libint](https://github.com/evaleev/libint)

# 0.8.0

### Fixed

* Passed to libint2 the Cartesian coordinates in Angstrom instead *atomic units*.


# 0.7.0

### New

 * Allow to compute charge molecules in the *C2Pk* input.
 * Compute the multipole integrals in the center of mass.
 * A new variable called ``aux_fix`` has been introduced to change the quality of the auxiliar basis set
   for hybrid calculation. The possible values are: "low", "medium", "good", "verygood" and "excellent".
   The default value is: verygood.
 * Return a ``input_parameters.yml`` file containing the input after all the preprocessing steps.

## Change

 * The ``path_basis`` variable in the yaml input, points to the folder where all the CP2K basis are located.
   By Default this variable points to <Installation>/nac/basis where there are some default basis.

### Deleted

* The ``path_potential`` variable has been removed since it is superseded by the ``path_basis``.


# 0.6.0

### New
 * Compute the overlap integrals to calculate the derivative coupling and the multipole integrals using [libint2](https://github.com/evaleev/)
 * Used `openmp` to compute the integrals in all the available cores
 * New dependencies: [eigen](http://eigen.tuxfamily.org/dox/), [highfive](https://github.com/BlueBrain/HighFive/tree/master/include/highfive), [libint2](https://github.com/evaleev/libint/wiki) and [pybind11](https://pybind11.readthedocs.io/en/master/)
 
### Deleted
 
 * Python/Cython implementation of the overlap integrals
 * Unused functionality replaced by [libint2](https://github.com/evaleev/)

# 0.5.0

### New

* The user only need to provide an **active_space** and both the `mo_index_range` and `nHOMO`  keywords are computed automatically.

* Added fast test to [compute the couplings](https://github.com/SCM-NV/nano-qmflows/blob/master/test/test_coupling.py)

### Deleted

* Removed all the Fourier transform for orbitals.

* Removed unused electron transfer functionality.

### Changed

* The `nHOMO` and the `kinds` for the *CP2K* input are computed using the [valence_electrons](https://github.com/SCM-NV/nano-qmflows/blob/master/nac/basisSet/valence_electrons.json) from the basis/pseudpotential combination.

* Use a configuration dictionary to around the initial input instead of many arguments functions.

* Import only the most important functions.


# 0.4.1

### Deleted

* Removed all the MPI unused functionality

### Changed

* Refactor the [distribute_jobs.py](https://github.com/SCM-NV/nano-qmflows/blob/master/scripts/distribution/distribute_jobs.py) script to split the derivative coupling calculations.

# 0.4.0

### Deleted

* removed `workflow_oscillator_strength`. Use `workflow_stddft` instead

### Changed

* Moved `nHomo` keyword to `general_setting`
* Renamed the `ci_range` keyword and replaced it by the *CP2K* keyword `mo_index_range`

### New
* Templates to call functionals **pbe** and **pbe0** to compute the Molecular orbitals


## 0.3.1

### Changed

* Replace the `json schemas` with the [schemas](https://github.com/keleshev/schema) library


## 0.3.0

### Added

The following actions were performed:
* Removed nose and pandas dependencies
* Use pytest for testing
* Replace MIT license by Apache-2.0
* Allow only fast tests in Travis
* Added changelog
* made general mergeHDF5 script
* Added Runners: MPI and Multiprocessing(default)
* Introduce new input file (yaml)
* Validate input files with json schemas
* Refactor the workflow API
* Used [noodles==0.3.1](https://github.com/NLeSC/noodles) and [qmflows==0.3.0](https://github.com/SCM-NV/qmflows)
   
   
### Removed

* Dead code from `workflow_cube`
