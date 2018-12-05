# Change Log


## 0.3.1

### Changed

* Replace the `json schemas` with the [schemas](https://github.com/keleshev/schema) library.


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
