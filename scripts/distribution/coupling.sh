#! /bin/bash

python - <<EOF
from workflow_coupling import process_arguments
process_arguments(
""" 
# High Level Input for NA-MD Simulations using QM package
# ######################
# General Keywords
# ######################
QM_package = "CP2K"
Project_name = "NAC"
Scratch_folder = "/scratch-shared"
Trajectory_path = "./data/Cd33Se33_PBE_MD_1000points.xyz"
NumberOfTrajectoryBlocks = 4
NumberOfNodesPerBlock = 2
NumberOfProcsPerBlock = 16 
# ######################
# QM Package Mandatory Keywords
# ######################
Basis_set_folder = "$HOME/cp2k_basis"
Potential_folder = "$HOME/cp2k_basis"
Basis_set = "DZVP-MOLOPT-SR-GTH"
Potential = "GTH-PBE"
Cell_parameters = [28.0, 28.0, 28.0]
Added_MOs = 100 # Keep this number not very high 
MO_Index_Range = "10 100" # Keep this number because the I/O is more efficient
"""
)
EOF

