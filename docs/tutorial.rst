Cp2k Input generation
#####################

A minimal cp2k_

    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [50.0] * 3
    cp2k_args.specific.cp2k.force_eval.dft.scf.added_mos = 100
    cp2k_args.specific.cp2k.force_eval.dft.scf.diagonalization.jacobi_threshold = 1e-6



qmworks_ automatically create the indented structure of the previous example together with the special character *&* at
the beginning and end of each section, and finally the keyword *END* at the end of each section.

    
Notice that *CP2K* requires the explicit declaration of the basis set together with the charge and the name of
the potential used for each one of the atoms. In the previous example the basis for the carbon is *DZVP-MOLOPT-SR-GTH*,
while the potential is *GTH-PBE* and the charge *q4*.Also, the simulation cell can be specified using the x, y, z vectors
like in the previous example. but also, a cubic box can be easily specified by: ::
  
  penta.input.force_eval.subsys.cell.ABC = "[angstrom] 50 50 50"

that result in a simulation cube of 50 cubic angstroms.

For a more detailed description of cp2k_ input see manual_.
    


WOrkFlow to compute PYXAID Hamiltonian
######################################


The function to calculate the Hamiltonian to carry out *Nonadiabatic molecular dynamics* using PYXAID_: ::

   def generate_pyxaid_hamiltonians(package_name, project_name, all_geometries,
                                    cp2k_args, guess_args=None,
                                    calc_new_wf_guess_on_points=[0],
                                    path_hdf5=None, enumerate_from=0,
                                    package_config=None):
       """
       Use a md trajectory to generate the hamiltonian components to tun PYXAID
       nmad.

       :param package_name: Name of the package to run the QM simulations.
       :type  package_name: String
       :param project_name: Folder name where the computations
       are going to be stored.
       :type project_name: String
       :param all_geometries: List of string cotaining the molecular geometries
       numerical results.
       :type path_traj_xyz: [String]
       :param package_args: Specific settings for the package
       :type package_args: dict
       :param use_wf_guess_each: number of Computations that used a previous
       calculation as guess for the wave function.
       :type use_wf_guess_each: Int
       :param enumerate_from: Number from where to start enumerating the folders
       create for each point in the MD
       :type enumerate_from: Int
       :param package_config: Parameters required by the Package.
       :type package_config: Dict
       :returns: None
       """

.. image:: docs/images/nac_worflow.png



.. _cp2k: https://www.cp2k.org/

.. _manual: https://manual.cp2k.org/#gsc.tab=0
