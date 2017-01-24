
##  <font color='green'> Installation</font>
In order to install the *nonadiabaticCoupling* library you need to install first the **QMWorks** package and its environment using *Anaconda* as detailed [here](https://github.com/SCM-NV/qmworks).

Then,  to install the `` nonadiabaticCoupling`` library type the following command inside the conda environment:
```
(qmworks) user@server> pip install https://github.com/felipeZ/nonAdiabaticCoupling/tarball/master#egg=qmworks --upgrade
```
Note: **QMWorks** is installed using [conda](http://conda.pydata.org/docs/index.html), we suggest you to use the same virtual environment to run the coupling calculations.

## <font color='green'> Nonadiabatic coupling matrix</font>
The total time-dependent wave function  $\Psi(\mathbf{R}, t)$ can be expressed in terms of a linear combination
of N adiabatic electronic eigenstates $\phi_{i}(\mathbf{R}(t))$,

 $$
\Psi(\mathbf{R}, t) =  \sum^{N}_{i=1} c_i(t)\phi_{i}(\mathbf{R}(t)) \quad \mathbf(1)
$$ 


The time-dependent coefficients are propagated according to

$$
\frac{dc_j(t)}{dt} = -i\hbar^2 c_j(t) E_j(t) - \sum^{N}_{i=1}c_i(t)\sigma_{ji}(t)  \quad \mathbf(2)
$$

where $E_j(t)$ is the energy of the jth adiabatic state and $\sigma_{ji}(t)$ the nonadiabatic matrix, which elements are given by the expression

$$
\sigma_{ji}(t) = \langle \phi_{j}(\mathbf{R}(t)) \mid \frac{\partial}{\partial t} \mid \phi_{i}(\mathbf{R}(t)) \rangle
 \quad \mathbf(3)
$$

that can be approximate using three consecutive molecular geometries

$$
\sigma_{ji}(t) \approx \frac{1}{4 \Delta t} (3\mathbf{S}_{ji}(t) - 3\mathbf{S}_{ij}(t) - \mathbf{S}_{ji}(t-\Delta t) + \mathbf{S}_{ij}(t-\Delta t))
 \quad \mathbf(4)
$$

where $\mathbf{S}_{ji}(t)$ is the overlap matrix between two consecutive time steps
$$
\mathbf{S}_{ij}(t) = \langle \phi_{j}(\mathbf{R}(t-\Delta t)) \mid \phi_{i}(\mathbf{R}(t)) \rangle
 \quad \mathbf(5)
$$

and the overlap matrix is calculated in terms of atomic orbitals
$$
\mathbf{S}_{ji}(t) = \sum_{\mu} C^{*}_{\mu i}(t) \sum_{\nu} C_{\nu j}(t - \Delta t) \mathbf{S}_{\mu \nu}(t)
 \quad \mathbf(6)
$$

Where 
$C_{\mu i}$ are the Molecular orbital coefficients and $\mathbf{S}_{\mu \nu}$ The atomic orbitals overlaps.

$$
  \mathbf{S}_{\mu \nu}(\mathbf{R}(t), \mathbf{R}(t - \Delta t)) = 
  \langle \chi_{\mu}(\mathbf{R}(t)) \mid \chi_{\nu}(\mathbf{R}(t - \Delta t)\rangle
 \quad \mathbf(7)
$$

### <font color='green'> Nonadiabatic coupling algorithm implementation</font>
<justify>The  figure belows shows schematically the workflow for calculating the Nonadiabatic 
coupling matrices from a molecular dynamic trajectory. The uppermost node represent a molecular dynamics trajectory that is subsequently divided in its components and  for each geometry the molecular orbitals are computed. These molecular orbitals are stored in a [HDF5](http://www.h5py.org/) binary file and subsequently calculations retrieve sets of three molecular orbitals that are use to calculate the nonadiabatic coupling matrix using equations **4** to **7**. These coupling matrices are them feed to the *[PYXAID](https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html)* package to carry out nonadiabatic molecular dynamics.</justify>

 ![title](files/nac_worflow.png)
 Workflow for the calculation of the Nonadiabatic coupling using CP2K

The Overlap between primitives are calculated using the Obara-Saika recursive scheme, that has been implemented as a [cython](http://cython.org) module for efficiency reasons. The nonadiabatic coupling module uses the aforementioned
module together with the [multiprocess](https://docs.python.org/3.6/library/multiprocessing.html) Python library to distribute the overlap matrix computations among the CPUs available. Also, all the heavy numerical processing is carried out by the highly optimized functions in [numpy](http://www.numpy.org).

 The **nonadiabaticCoupling** package relies on *QMWorks* to run the Quantum mechanical simulations using the [CP2K](https://www.cp2k.org/) package.  Also, the [noodles](library) is used  to schedule expensive numerical computations that are required to calculate the nonadiabatic coupling matrix.



# <font color='green'> Running the workflow</font>

There are 2 steps to compute the nondiabatic couplings for a given molecular dynamics: 
    1. Create the scripts to perform the simulation
    2. Running the scripts
    
Below is  shown the script responsible for the first step of the simulation, which is available at: https://github.com/felipeZ/nonAdiabaticCoupling/blob/master/scripts/distribution/distribute_jobs.py

This script takes as input a minimal slurm configuration that can be modified by the user. The slurm configuration
contains the number of nodes and task, together with the walltime requested and the job name. In the table below the default Slurm values for these parameters is shown


 | property | default |
 |:--------:|:-------:|
 | nodes    |    2    |
 | tasks    |    24   |
 | time     | 48:00:00|
 | name     | namd    |


Also, the user should provided the following parameters for the simulation

    * path where the CP2K calculation will be created (``scratch``)
    * project_name
    * path to the basis and Cp2k Potential
    * CP2K parameters:
      - Range of Molecular oribtals printed by CP2K
      - Cell parameter
    * Settings to Run Cp2k simulations
    * Path to the trajectory in XYZ format


This is the script corresponding to the step 1:

```python
from collections import namedtuple
from nac.workflows.initialization import split_trajectory
from os.path import join
from qmworks import Settings
from qmworks.utils import settings2Dict

import getpass
import os
import shutil
import string
import subprocess

SLURM = namedtuple("SLURM", ("nodes", "tasks", "time", "name"))


def main():
    """
    THE USER MUST CHANGES THESE VARIABLES ACCORDING TO HER/HIS NEEDS:
      * project_name
      * path to the basis and Cp2k Potential
      * blocks (number of batches into which the trajectory is splitted)
      * CP2K:
          - Range of Molecular oribtals printed by CP2K
          - Cell parameter
      * Settings to Run Cp2k simulations
      * Path to the trajectory in XYZ

    The slurm configuration is optional but the user can edit it:
        property  default
       * nodes         2
       * tasks        24
       * time   48:00:00
       * name       namd

    """
    # USER DEFINED CONFIGURATION
    scratch = 'scratch-shared/user29/jobs_quantumdot'
    project_name = 'Quantumdot'  # name used to create folders

    # Path to the basis set used by Cp2k
    home = os.path.expanduser('~')
    basisCP2K = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potCP2K = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    lower_orbital, upper_orbital = 278, 317
    cp2k_main, cp2k_guess = cp2k_input(lower_orbital, upper_orbital,
                                       cell_parameters=28)

    # Trajectory splitting
    path_to_trajectory = "traj1000.xyz"
    blocks = 5  # Number of chunks to split the trajectory

    # SLURM Configuration
    slurm = SLURM(
        nodes=2,
        tasks=24,
        time="48:00:00",
        name="namd"
    )

    distribute_computations(project_name, basisCP2K, potCP2K, cp2k_main,
                            cp2k_guess, path_to_trajectory, blocks, slurm)


def cp2k_input(lower_orbital, upper_orbital, cell_parameters=None):
    """
    # create ``Settings`` for the Cp2K Jobs.
    """
    # Main Cp2k Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [cell_parameters] * 3
    main_dft = cp2k_args.specific.cp2k.force_eval.dft
    main_dft.scf.added_mos = 20
    main_dft.scf.max_scf = 200
    main_dft.scf.eps_scf = 1e-5
    main_dft['print']['mo']['mo_index_range'] = "{} {}".format(lower_orbital,
                                                               upper_orbital)
    cp2k_args.specific.cp2k.force_eval.subsys.cell.periodic = 'None'

    # Setting to calculate the wave function used as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_OT.potential = "GTH-PBE"
    cp2k_OT.cell_parameters = [cell_parameters] * 3
    ot_dft = cp2k_OT.specific.cp2k.force_eval.dft
    ot_dft.scf.scf_guess = 'atomic'
    ot_dft.scf.ot.minimizer = 'DIIS'
    ot_dft.scf.ot.n_diis = 7
    ot_dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    ot_dft.scf.added_mos = 0
    ot_dft.scf.eps_scf = 1e-05
    ot_dft.scf.scf_guess = 'restart'
    cp2k_OT.specific.cp2k.force_eval.subsys.cell.periodic = 'None'

    return cp2k_args, cp2k_OT

# End of the user serviceable code
```

In the ``cp2k_input``  function the ``Settings`` to perform a single point calculation with CP2K are defined. Using this configuration [QMWorks](https://github.com/SCM-NV/qmworks) automatically create the CP2K input. You do not need to add the *&* or *&END* symbols, *QWorks* adds them automatically for you.

Notice that *CP2K* requires the explicit declaration of the basis set together with the name of the potential used for each one of the atoms. In the previous example the basis for the carbon is *DZVP-MOLOPT-SR-GTH*, while the potential is *GTH-PBE*. Also, the simulation cell can be specified using the x, y, z vectors (it this case a cubic cell is built).


### <font color='green'> Workflow distribution</font>
Once you fill in the required parameters you just need to run the script like:
```bash
user@server> python distribute_jobs.py
```
You will see that several folders were create: ``chunk_a``, ``chunk_b``, etc., where the number of files correspond with the number of block that you request in the script. The content of each folder is something similar to:
```bash
[user@server]$ ls chunk_a
chunk_xyz_a  launch.sh  script_remote_function.py
```

Each folder containts a ``chunk_x`` file containing molecular geometries in `xyz` format, a `slurm` ``launch.sh`` script file and a python script ``script_remote_function.py`` to compute the couplings.

**You only need to run the slurm script in order to compute the jobs.**

## <font color='green'> Restarting a Job </font>

Both the *molecular orbitals* and the *derivative couplings* for a given molecular dynamic trajectory are stored in a [HDF5](http://www.h5py.org/) file. The library check wether the *MO* orbitals or the coupling under consideration are already present in the ``HDF5`` file, otherwise compute it. Therefore  if the workflow computation fails due to a recoverable issue like:
    * Cancelation due to time limit.
    * Manual suspension or cancelation for another reasons.

Then, in order to restart the job you need to perform the following actions:
    * Remove the file called ``cache.json`` from the current work  directory.
    * Remove the plams folder  where the `CP2K` computation were carried out.
    
The `plams` folder is create inside the `scratch` folder that you have defined in the script to distribute the computation. In the previous example it was:
```python
 scratch = 'scratch-shared/user29/jobs_quantumdot'
 ``` 

## <font color='green'> Known Issues </font>

#### <font color='green'> Coupling distribution in multiple nodes </font>
`CP2K` can uses multiple nodes to perform the computation of the molecular orbitals using the [MPI](http://www.mcs.anl.gov/research/projects/mpi/) protocol. Unfortunately, we have not implemented `MPI` for the computation of the *derivative coupling matrix*. The practical consequences of the aforemention issues, is that **the 
calculation of the coupling matrices are carried out in only 1 computational node**. It means that if you want ask
for more than 1 node to compute the molecular orbitals with `CP2K`, once the workflow starts to compute the *derivative couplings* only 1 node will be used at a time and the rest will remain idle wating computational resources. 

#### <font color='green'> Memory allocation </font>
The *derivative couplings* computations are started once all the molecular orbitals have been calculated. Then, all the coupling calculation are scheduled, holding in memory all the molecular orbitals until they are requested.  It cause a huge memory consumption. 

#### <font color='green'> Reporting a bug or requesting a feature </font>
** To report  an issue or request a new feature you can use the issues tracker of  [github](https://github.com/felipeZ/nonAdiabaticCoupling/issues)**


## <font color='green'> The Coupling Workflow</font>
The following function is called by the ``script_remote_function.py`` to compute the molecular orbitals and the correspoding derivative couplings.


```python
def generate_pyxaid_hamiltonians(package_name, project_name,
                                 cp2k_args, guess_args=None,
                                 path=None,
                                 geometries=None, dictCGFs=None,
                                 calc_new_wf_guess_on_points=None,
                                 path_hdf5=None, enumerate_from=0,
                                 package_config=None, nCouplings=None,
                                 traj_folders=None, work_dir=None,
                                 basisname=None, hdf5_trans_mtx=None,
                                 dt=1):
    # prepare Cp2k Jobs
    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(package_name, geometries, project_name,
                                  path_hdf5, traj_folders, cp2k_args,
                                  guess_args, calc_new_wf_guess_on_points,
                                  enumerate_from,
                                  package_config=package_config)

    # Calculate Non-Adiabatic Coupling
    # Number of Coupling points calculated with the MD trajectory
    nPoints = len(geometries) - 2
    promise_couplings = [calculate_coupling(i, path_hdf5, dictCGFs,
                                            geometries,
                                            mo_paths_hdf5, hdf5_trans_mtx,
                                            enumerate_from,
                                            output_folder=project_name, dt=dt,
                                            nCouplings=nCouplings,
                                            units='angstrom')
                         for i in range(nPoints)]
    path_couplings = gather(*promise_couplings)

    # Write the results in PYXAID format
    path_hamiltonians = join(work_dir, 'hamiltonians')
    if not os.path.exists(path_hamiltonians):
        os.makedirs(path_hamiltonians)

    # Inplace scheduling of write_hamiltonians function.
    # Equivalent to add @schedule on top of the function
    schedule_write_ham = schedule(write_hamiltonians)

    promise_files = schedule_write_ham(path_hdf5, work_dir, mo_paths_hdf5,
                                       path_couplings, nPoints,
                                       path_dir_results=path_hamiltonians,
                                       enumerate_from=enumerate_from,
                                       nCouplings=nCouplings)

    hams_files = run(promise_files, path=path)

    print(hams_files)

```
