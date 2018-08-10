import h5py
from nac.basisSet import create_dict_CGFs
from qmflows.parsers.xyzParser import readXYZ
from nac.common import change_mol_units

import numpy as np 
import subprocess 
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist


def hardness(s: str):
    d = {'h': 6.4299, 'he': 12.5449, 'li': 2.3746, 'be': 3.4968, 'b': 4.619, 'c': 5.7410, 'n': 6.6824, 'o': 7.9854,
         'f': 9.1065, 'ne': 10.2303, 'na': 2.4441, 'mg': 3.0146, 'al': 3.5849, 'si': 4.1551, 'p': 4.7258,
         's': 5.2960, 'cl': 5.8662, 'ar': 6.4366, 'k': 2.3273, 'ca': 2.7587, 'sc': 2.8582, 'ti': 2.9578,
         'v': 3.0573, 'cr': 3.1567, 'mn': 3.2564, 'fe': 3.3559, 'co': 3.4556, 'ni': 3.555, 'cu': 3.6544,
         'zn': 3.7542, 'ga': 4.1855, 'ge': 4.6166, 'as': 5.0662, 'se': 5.4795, 'br': 5.9111, 'kr': 6.3418,
         'rb': 2.1204, 'sr': 2.5374, 'y': 2.6335, 'zr': 2.7297, 'nb': 2.8260, 'mo': 2.9221, 'tc': 3.0184,
         'ru': 3.1146, 'rh': 3.2107, 'pd': 3.3069, 'ag': 3.4032, 'cd': 3.4994, 'in': 3.9164, 'sn': 4.3332,
         'sb': 4.7501, 'te': 5.167, 'i': 5.5839, 'xe': 6.0009, 'cs': 0.6829, 'ba': 0.9201, 'la': 1.1571,
         'ce': 1.3943, 'pr': 1.6315, 'nd': 1.8686, 'pm': 2.1056, 'sm': 2.3427, 'eu': 2.5798, 'gd': 2.8170,
         'tb': 3.0540, 'dy': 3.2912, 'ho': 3.5283, 'er': 3.7655, 'tm': 4.0026, 'yb': 4.2395, 'lu': 4.4766,
         'hf': 4.7065, 'ta': 4.9508, 'w': 5.1879, 're': 5.4256, 'os': 5.6619, 'ir': 5.900, 'pt': 6.1367,
         'au': 6.3741, 'hg': 6.6103, 'tl': 1.7043, 'pb': 1.9435, 'bi': 2.1785, 'po': 2.4158, 'at': 2.6528,
         'rn': 2.8899, 'fr': 0.9882, 'ra': 1.2819, 'ac': 1.3497, 'th': 1.4175, 'pa': 1.9368, 'u': 2.2305,
         'np': 2.5241, 'pu': 3.0436, 'am': 3.4169, 'cm': 3.4050, 'bk': 3.9244, 'cf': 4.2181, 'es': 4.5116,
         'fm': 4.8051, 'md': 5.0100, 'no': 5.3926, 'lr': 5.4607 }
    return d[s]

def n_sph(s: str):
    d = {'cd': 25, 'se': 13}
    return d[s] 

# Some basic input variables
project_name = 'Cd33Se33_QD'
path_hdf5 = 'Cd33Se33.hdf5' 
basisname='DZVP-MOLOPT-SR-GTH'
path_overlap = 'overlap_Cd33Se33_cp2k.npy'
path_tdm = 'tdm_Cd33Se33_cp2k.npy'
mol_file = 'Cd33Se33.xyz'

# Some basic input variable for the sTDA calculations
ax = 0.0 # For PBE0 . It changes depending on the functional. A dictionary should be written to store these values. 
alpha1 = 1.42 # These values are fitted by Grimme (2013)   
alpha2 = 0.48 # These values are fitted by Grimme (2013)
beta1 = 0.2 # These values are fitted by Grimme (2013)
beta2 = 1.83 # These values are fitted by Grimme (2013)


# Loading main stuff
f5 = h5py.File(path_hdf5, 'r') # Open the hdf5 file with MOs and energy values 
c_ao = f5['{}/point_0/cp2k/mo/coefficients'.format(project_name)].value # Read MOs coefficients in AO basis. Matrix size: NAO x NMO 
e = f5['{}/point_0/cp2k/mo/eigenvalues'.format(project_name)].value
s = np.load(path_overlap) # Load Overlap matrix in AO basis 
tdm = np.load(path_tdm) # Load transition dipole moment matrix in AO basis 
#mol = change_mol_units(readXYZ(mol_file)) # Load molecule in Angstroms and convert it in atomic units
n_atoms, n_frames = get_numberofatoms(mol_file) 
atoms = read_atomlist(mol_file, n_atoms)
nocc = 50 # Number of occupied orbitals
nvirt = c_ao.shape[1] - nocc # Number of virtual orbitals 

# Create a dictionary to assign the number of spherical functions for each atom. We will need it later in several places.  
#dictCGFs = create_dict_CGFs(path_hdf5, basisname, mol)
#cgfs_per_atoms = {s: len(dictCGFs[s]) for s in dictCGFs.keys()}
# There should be a better way to know the number of spherical functions. Now hard-coded. 

# Now compute the charge transition densities using the Lowdin formula  
# First transform the c_ao in MO basis
sqrt_s = sqrtm(s)
c_mo = np.dot(sqrt_s, c_ao)
q = np.zeros((n_atoms, c_mo.shape[1], c_mo.shape[1])) # Size of the transition density tensor : n_atoms x n_mos x n_mos 

index = 0 
for i in range(n_atoms):
    q[i, :, :] = np.dot(c_mo[index:(index + n_sph(atoms[i])), :].T, c_mo[index:(index + n_sph(atoms[i])), :])
    index += n_sph(atoms[i])

# Compute the Mataga-Nishimoto-Ohno_Klopman damped Columb and Excgange law functions 
coords = read_xyz_coordinates(mol_file, n_atoms, 1)
coords = coords * 1.8897259886 # Conversion from Angstroms to atomic units
r_ab = cdist(coords, coords) # Distance matrix between atoms A and B 
hardness_vec = np.stack(hardness(atoms[i]) for i in range(n_atoms)).reshape(n_atoms, 1)
hard = np.dot(hardness_vec, hardness_vec.T) / 2 

beta = beta1 + ax * beta2
alpha = alpha1 + ax * alpha2 

gamma_J = np.power(1 / (np.power(r_ab, beta) + ax * np.power(hard, -beta)), 1/beta)
gamma_J[gamma_J == np.inf] = 0 # When ax = 0 , you can get infinite values on the diagonal. Just turn them off to 0. 
gamma_K = np.power(1 / (np.power(r_ab, alpha) + np.power(hard, -alpha)), 1/alpha)

# Compute the Couloumb and Exchange integrals
pqrs_J = np.tensordot(q, np.tensordot(q, gamma_J, axes=(0, 1)), axes=(0, 2))
pqrs_K = np.tensordot(q, np.tensordot(q, gamma_K, axes=(0, 1)), axes=(0, 2))

# Construct the Tamm-Dancoff matrix A for each pair of i->a transition
# First step is to build the single excited Slater determinanes from one-electron orbitals  
excs = []
for i in range(nocc):
    for a in range(nocc, nvirt+nocc):
        excs.append((i,a))

# Now fill the A matrix 
k_iajb = 2 * pqrs_K[:nocc, nocc:, :nocc, nocc:].reshape(nocc*nvirt, nocc*nvirt) # This is the exchange integral entering the A matrix. It is in the format (nocc, nvirt, nocc, nvirt)
k_ijab = ax * pqrs_J[:nocc, :nocc, nocc:, nocc:] # This is the Coulomb integral entering in the A matrix. It is in the format: (nocc, nocc, nvirt, nvirt)
k_ijab = np.swapaxes(pqrs_J_tmp, axis1=1, axis2=2).reshape(nocc*nvirt, nocc*nvirt) # To get the correct order in the A matrix, i.e. (nocc, nvirt, nocc, nvirt), we have to swap axes 

a_mat = k_iajb - k_ijab # They are in the m x m format where m is the number of excitations = nocc * nvirt  

e_diff = -np.subtract(e[:nocc].reshape(nocc,1) , e[nocc:].reshape(nvirt, 1).T).reshape(nocc*nvirt) # Generate a vector with all possible ea - ei energy differences 
np.fill_diagonal(a_mat, np.diag(a_mat) + e_diff)

#a_mat = np.zeros((nocc*nvirt, nocc*nvirt))
#e_mat = np.zeros((nocc*nvirt, nocc*nvirt))
#for I in range(len(excs)):
#    for J in range(len(excs)):
#        a_mat[I, J] = 2 * pqrs_K[excs[I][0], excs[I][1], excs[J][0], excs[J][1]] - ax * pqrs_J[excs[I][0], excs[J][0], excs[I][1], excs[J][1]]
#        if excs[I][0] == excs[J][0] and excs[I][1] == excs[J][1]:
#            a_mat[I, J] += e[excs[I][1]] - e[excs[I][0]]
#            e_mat[I, J] = e[excs[I][1]] - e[excs[I][0]]


# Solve the eigenvalue problem = A * cis = omega * cis 
omega, cis = np.linalg.eig(a_mat)

# Compute transition dipole moments
#pre_factor = np.empty(nocc*nvirt)
#pre_factor = np.hstack(np.sqrt(2 * e_diff/omega[i]) for i in range(nocc*nvirt)).reshape(nocc*nvirt, nocc, nvirt)

pre_factor = np.sqrt( 2 * np.divide(e_diff.reshape(nocc*nvirt, 1), omega.reshape(1, nocc*nvirt))).T.reshape(nocc*nvirt, nocc, nvirt)
cis_new = cis.reshape(nocc, nvirt, nocc*nvirt)
tdmatrix_x = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[0, :, :], c_ao[:, nocc:]])
tdmatrix_y = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[1, :, :], c_ao[:, nocc:]])
tdmatrix_z = np.linalg.multi_dot([c_ao[:, :nocc].T, tdm[2, :, :], c_ao[:, nocc:]])

d_x = np.hstack(np.trace(np.linalg.multi_dot([pre_factor[i, :, :], cis_new[:, :, i].T, tdmatrix_x])) for i in range(nocc*nvirt))
d_y = np.hstack(np.trace(np.linalg.multi_dot([pre_factor[i, :, :], cis_new[:, :, i].T, tdmatrix_y])) for i in range(nocc*nvirt))
d_z = np.hstack(np.trace(np.linalg.multi_dot([pre_factor[i, :, :], cis_new[:, :, i].T, tdmatrix_z])) for i in range(nocc*nvirt))

f = 2 / 3 * omega * (d_x **2 + d_y ** 2 + d_z ** 2)

# Write to output 

# Retrieve some useful information from data
weight = np.hstack(np.max(cis[:, i] ** 2) for i in range(nocc*nvirt)) # weight of the most important transition for an excited state
index_weight = np.hstack(np.where( cis[:, i] ** 2 == np.max(cis[:, i] ** 2) ) for i in range(nocc*nvirt)).reshape(nocc*nvirt) # Find the index of this transition
index_i = np.stack(excs[index_weight[i]][0] for i in range(nocc*nvirt)) # Index of the hole
index_a = np.stack(excs[index_weight[i]][1] for i in range(nocc*nvirt))
e_orb_i = e[index_i] * 27.211 # These are the energies of the hole for the transition with the larger weight
e_orb_a = e[index_a] * 27.211  # These are the energies of the electron for the transition with the larger weight
e_singorb = (e_orb_a - e_orb_i )  # This is the energy for the transition with the larger weight 

output = np.empty((nocc*nvirt, 12))
output[:, 0] = 0 # State number: we update it after reorder
output[:, 1] = omega * 27.211 # State energy in eV
output[:, 2] = f # Oscillator strength
output[:, 3] = d_x # Transition dipole moment in the x direction
output[:, 4] = d_y # Transition dipole moment in the y direction
output[:, 5] = d_z # Transition dipole moment in the z direction
output[:, 6] = weight # Weight of the most important excitation
output[:, 7] = index_i # Index of the hole for the most important excitation
output[:, 8] = e_orb_i # hole energy
output[:, 9] = index_a 
output[:, 10] = e_orb_a
output[:, 11] = e_singorb 

output = output[output[:, 1].argsort()] # Reorder the output in ascending order of energy 
output[:, 0] = np.arange(nocc * nvirt) # Give a state number in the correct order

np.savetxt('output.txt', output, fmt='%5d %10.3f %10.5f %10.5f %10.5f %10.5f %10.5f %3d %10.3f %3d %10.3f %10.3f')




def get_numberofatoms(fn): 
    # Retrieve number of lines in trajectory file
    cmd = "wc -l {}".format(fn)
    l = subprocess.check_output(cmd.split()).decode()
    n_lines = int(l.split()[0]) # Number of lines in traj file

    # Read number of atoms in the molecule. It is usually the first line in a xyz file  
    with open(fn) as f:
       l = f.readline()
       n_atoms = int(l.split()[0])

    # Get the number of frames in the trajectory file
    n_frames = int(int(n_lines)/(n_atoms+2))
    return n_atoms, n_frames 

def read_atomlist(fn, n_atoms):
    # Read atomic list from xyz file
    atoms = pd.read_csv(fn, nrows = n_atoms, delim_whitespace=True, header=None, 
                        skiprows = 2, usecols=[0]).astype(str)
    return atoms[0][:]      


def read_xyz_coordinates(fn, n_atoms, iframe): 
    # Read xyz coordinate from a (trajectory) xyz file. 
    coords = pd.read_csv(fn, nrows = n_atoms, delim_whitespace=True, header=None, 
             skiprows = (2 + (n_atoms + 2) * (iframe - 1)), usecols=(1,2,3)).astype(float).values
    return coords  


