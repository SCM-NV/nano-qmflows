# Change Log

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

* Added fast test to [compute the couplings](https://github.com/SCM-NV/qmflows-namd/blob/master/test/test_coupling.py)

### Deleted

* Removed all the Fourier transform for orbitals.

* Removed unused electron transfer functionality.

### Changed

* The `nHOMO` and the `kinds` for the *CP2K* input are computed using the [valence_electrons](https://github.com/SCM-NV/qmflows-namd/blob/master/nac/basisSet/valence_electrons.json) from the basis/pseudpotential combination.

* Use a configuration dictionary to around the initial input instead of many arguments functions.

* Import only the most important functions.


# 0.4.1

### Deleted

* Removed all the MPI unused functionality

### Changed

* Refactor the [distribute_jobs.py](https://github.com/SCM-NV/qmflows-namd/blob/master/scripts/distribution/distribute_jobs.py) script to split the derivative coupling calculations.

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
