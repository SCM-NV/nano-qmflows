# Change Log

# 0.4.1

### Deleted

* Remove all the MPI unused functionality

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
