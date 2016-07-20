
# ================> Python Standard  and third-party <==========
from os.path import join

# ==============================> Main <==================================


def main():
    """
    Initialize the arguments to compute the nonadiabatic coupling matrix for
    a given MD trajectory.
    """
    plams.init()
    project_name = 'ET_Pb79S44'

    # create Settings for the Cp2K Jobs
    cp2k_args = Settings()
    cp2k_args.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_args.potential = "GTH-PBE"
    cp2k_args.cell_parameters = [50.0] * 3
    cp2k_args.specific.cp2k.force_eval.dft.scf.added_mos = 100
    cp2k_args.specific.cp2k.force_eval.dft.scf.diagonalization.jacobi_threshold = 1e-6

    # Setting to calculate the WF use as guess
    cp2k_OT = Settings()
    cp2k_OT.basis = "DZVP-MOLOPT-SR-GTH"
    cp2k_OT.potential = "GTH-PBE"
    cp2k_OT.cell_parameters = [50.0] * 3
    cp2k_OT.specific.cp2k.force_eval.dft.scf.scf_guess = 'atomic'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.minimizer = 'DIIS'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.n_diis = 7
    cp2k_OT.specific.cp2k.force_eval.dft.scf.ot.preconditioner = 'FULL_SINGLE_INVERSE'
    cp2k_OT.specific.cp2k.force_eval.dft.scf.added_mos = 0
    cp2k_OT.specific.cp2k.force_eval.dft.scf.eps_scf = 5e-06

    # Path to the MD geometries
    path_traj_xyz = "./trajectory_4000-5000.xyz"

    # User variables
    home = os.path.expanduser('~')  # HOME Path
    username = getpass.getuser()
    
    # Work_dir
    scratch = "/scratch-shared"
    scratch_path = join(scratch, username, project_name)
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)

    # Cp2k configuration files
    basiscp2k = join(home, "Cp2k/cp2k_basis/BASIS_MOLOPT")
    potcp2k = join(home, "Cp2k/cp2k_basis/GTH_POTENTIALS")
    cp2k_config = {"basis": basiscp2k, "potential": potcp2k}

    # HDF5 path
    path_hdf5 = join(scratch_path, 'quantum.hdf5')

    # all_geometries type :: [String]
    geometries = split_file_geometries(path_traj_xyz)

    # Named the points of the MD starting from this number
    enumerate_from = 0

    # Calculate new Guess in each Geometry
    pointsGuess = [enumerate_from + i for i in range(len(geometries))]

    # Dynamics time step in Femtoseconds
    dt = 1
    
    # Hamiltonian computation
    generate_pyxaid_hamiltonians('cp2k', project_name, geometries, cp2k_args,
                                 guess_args=cp2k_OT,
                                 calc_new_wf_guess_on_points=pointsGuess,
                                 path_hdf5=path_hdf5,
                                 enumerate_from=0,
                                 package_config=cp2k_config, dt=dt)

    print("PATH TO HDF5:{}\n".format(path_hdf5))
    plams.finish()
    
# ===================================<>========================================
if __name__ == "__main__":
    main()
