import h5py
from nac.basisSet import create_dict_CGFs
from qmflows.parsers.xyzParser import readXYZ
from nac.common import change_mol_units

import numpy as np 
import suprocess 
from scipy.linalg import sqrtm

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
    d = {'cd': 25, 'se': 14}
    return d[s] 

# Some basic input variables
project_name = 'Cd33Se33_QD'
path_hdf5 = 'Cd33Se33.hdf5' 
basisname='DZVP-MOLOPT-SR-GTH'
path_overlap = 'overlap_Cd33Se33_cp2k.npy'
mol_file = 'Cd33Se33.xyz'

# Some basic input variable for the sTDA calculations
ax = 0.25 # For PBE0 . It changes depending on the functional. A dictionary should be written to store these values. 
alpha1 = 1.42 # These values are fitted by Grimme (2013)   
alpha2 = 0.48 # These values are fitted by Grimme (2013)
beta1 = 0.2 # These values are fitted by Grimme (2013)
beta2 = 1.83 # These values are fitted by Grimme (2013)

# Loading main stuff
f5 = h5py.File(path_hdf5, 'r') # Open the hdf5 file with MOs and energy values 
c_ao = f5['{}/point_0/cp2k/mo/coefficients'.format(project_name)].value # Read MOs coefficients in AO basis. Matrix size: NAO x NMO 
e = f5['{}/point_0/cp2k/mo/eigenvalues'.format(project_name)].value
s = np.load(path_overlap) # Load Overlap matrix in AO basis 
#mol = change_mol_units(readXYZ(mol_file)) # Load molecule in Angstroms and convert it in atomic units
n_atoms, n_frames = get_numberofatoms(mol_file) 
atoms = read_atomlist(mol_file, n_atoms)

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
    q[i, :, :] = np.dot(c_mo[index:(index + n_sph(np.asscalar(atoms[i]))), :].T, c_mo[index:(index + n_sph(np.asscalar(atoms[i]))), :])
    index += n_sph(mol[i].symbol)

coords = read_xyz_coordinates(fn, n_atoms, 1)
r_ab = make_bond_matrix_f(coords, n_atoms)

# Compute the Mataga-Nishimoto-Ohno_Klopman damped Columb and Excgange law functions 
beta = beta1 + ax * beta2
alpha = alpha1 + ax * alpha2 
gamma_J = np.empty((n_atoms, n_atoms))
gamma_K = np.empty((n_atoms, n_atoms))

for b in range(n_atoms):
    for a in range(n_atoms):
        gamma_J[a, b] = np.power( 1 / ( np.power(r_ab[a, b], beta) + np.power((ax * hardness(np.asscalar(atoms[a])) * hardness(np.asscalar(atoms[b])) / 2), -beta)), 1/beta)
        gamma_K[a, b] = np.power( 1 / ( np.power(r_ab[a, b], alpha) + np.power(hardness(np.asscalar(atoms[a])) * hardness(np.asscalar(atoms[b])) / 2, -alpha)), 1/alpha)

# Compute the integrals

pqrs_J = np.tensordot(q.T, np.tensordot(gamma_J, q, axes =1), axes=1)
pqrs_K = np.tensordot(q.T, np.tensordot(gamma_K, q, axes =1), axes=1)

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
                        skiprows = 2, usecols=[0]).astype(str).values
    return atoms     


def read_xyz_coordinates(fn, n_atoms, iframe): 
    # Read xyz coordinate from a (trajectory) xyz file. 
    coords = pd.read_csv(fn, nrows = n_atoms, delim_whitespace=True, header=None, 
             skiprows = (2 + (n_atoms + 2) * (iframe - 1)), usecols=(1,2,3)).astype(float).values
    
    return coords  







