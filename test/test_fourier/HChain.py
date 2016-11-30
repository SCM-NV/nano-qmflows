
import sys
sys.path.append('/home/prokop/git_SW/nonAdiabaticCoupling')

data_dir = "/home/prokop/Desktop/kscan_qmworks/HChain"

from functools import (partial, reduce)
from math import (pi, sqrt)
from multiprocessing import Pool
from nac.integrals.fourierTransform import calculate_fourier_trasform_cartesian, calculate_fourier_trasform_cartesian_prokop, get_fourier_basis
from nac.schedule.components        import create_dict_CGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ
from qmworks.utils import concat

from nac import retrieve_hdf5_data

import h5py

import matplotlib.pyplot as plt

import argparse
import itertools
import numpy as np
import os

# Some Hint about the types
from typing import (Callable, List, Tuple)
Vector = np.ndarray
Matrix = np.ndarray




def projectionsToBins( kdata, Es, Emin=-6.0, Emax=0.0, dE=0.02 ):
    nks = kdata.shape[1]
    nE   = int((Emax-Emin)/dE) 
    bins = np.zeros( (nE, nks) )
    for i,Ei in enumerate(Es):
        iE = int((Ei-Emin)/dE)
        print( i, iE, Ei, nE, nks )
        if (iE>0) and (iE<nE):
            bins[iE,:] = np.maximum( bins[iE,:], kdata[i,:] )
    extent=(0,10,Emin,Emax)
    return bins, extent



def print_attrs(name, obj):
    print( name )
    #for key, val in obj.attrs.iteritems():
    #    print( "    %s: %s" % (key, val) )









def main(parser):
    """
    These calculation is based on the paper:
    `Theoretical analysis of electronic band structure of
    2- to 3-nm Si nanocrystals`
    PHYSICAL REVIEW B 87, 195420 (2013)
    """
   
    # Parse Command line
    project_name, path_hdf5, path_xyz, lattice_cte, basis_name, lower, \
        upper = read_cmd_line(parser)
    # Coordinates transformation
    path_hdf5 = join( data_dir, path_hdf5 )
    path_xyz  = join( data_dir, path_xyz )
    atoms     = readXYZ(path_xyz)
    symbols         = np.array([at.symbol for at in atoms])
    coords_angstrom = np.array([at.xyz    for at in atoms])
    
    #atoms = np.array(atoms)
    print( coords_angstrom )
    #f = h5py.File(path_hdf5,'r')
    #f.visititems(print_attrs)
    
    path_energy = project_name+'/point_0/cp2k/mo/eigenvalues'
    path_coefs  = project_name+'/point_0/cp2k/mo/coefficients'
    Es = retrieve_hdf5_data( path_hdf5, [path_energy])[0]
    print( len(Es) )
    print( np.array(Es)*27.2114 )
    
    ''''
    coefs = retrieve_hdf5_data( path_hdf5, [path_coefs])[0]
    print( coefs )
    print( coefs.shape )
    for i,coef in enumerate(coefs):
        plt.plot( coords_angstrom[:,0], coef )
    
    plt.show()
    exit()
    '''
    
    symbols = np.array([at.symbol for at in atoms])
    coords_angstrom = np.concatenate([at.xyz for at in atoms])
    angstroms_to_au = 1.889725989
    coords = angstroms_to_au * coords_angstrom
    lattice_cte = lattice_cte * angstroms_to_au

    # Dictionary containing as key the atomic symbols and as values the set of CGFs
    dictCGFs        = create_dict_CGFs(path_hdf5, basis_name, atoms)
    count_cgfs      = np.vectorize(lambda s: len(dictCGFs[s]))
    number_of_basis = np.sum(np.apply_along_axis(count_cgfs, 0, symbols))

    # K-space grid to calculate the fuzzy band
    nPoints = 100
    clin    = np.linspace(0.0,1.0,nPoints)[:,None]
    kmin    = np.array([-1.0,0.0,0.0]) 
    kmax    = np.array([+1.0,0.0,0.0])
    kpoints = kmin[None,:]*(1-clin) + clin*kmax[None,:]
    print("kpoints = ", kpoints)

    # Apply the whole fourier transform to the subset of the grid
    # correspoding to each process
    momentum_density = partial(compute_momentum_density, project_name, symbols,
                               coords, dictCGFs, number_of_basis, path_hdf5)

    orbitals = list(range(lower, upper + 1))
    dim_x    = len(orbitals)
    result   = np.empty((dim_x, nPoints))

    # Prokop Hapala V2 - not yet working
    print ("building basiset fourier dictionary ... ")
    chikdic = get_fourier_basis( symbols, dictCGFs, kpoints )
    print ("...fourier basis DONE !")
    for i, orb in enumerate(orbitals):
        print("Orbital: ", orb)
        orbK      = calculate_fourier_trasform_cartesian_prokop( symbols, coords, dictCGFs, number_of_basis, path_hdf5, project_name, orb, kpoints, chikdic=chikdic )
        orbKdens  = np.absolute( orbK )
        result[i] = orbKdens
        #print( orbKdens )
        #plt.plot( kpoints[:,0], orbKdens )
        
    Es = [ Es[i]*27.2114 for i in orbitals]
    bins, extent = projectionsToBins( result, Es, Emin=-15.0, Emax=5.0, dE=0.1 )
    
    plt.figure(figsize=(5,8))
    #plt.imshow( np.log10(bins), interpolation='nearest', origin='image', extent=extent, cmap='jet' )
    plt.imshow( bins, interpolation='nearest', origin='image', extent=extent, cmap='jet' )
    plt.colorbar()
    #plt.savefig( fname+".png", bbox='tight' )
    plt.show()    
    
    

def compute_momentum_density(project_name, symbols, coords, dictCGFs,
                             number_of_basis, path_hdf5, orbital):
    """
    Compute the reciprocal space density for a given Molecular
    Orbital.
    """
    
    # Compute the fourier transformation in cartesian coordinates
    fun_fourier = partial(calculate_fourier_trasform_cartesian, symbols, coords, dictCGFs, number_of_basis, path_hdf5, project_name, orbital)
    #fun_fourier  = partial(calculate_fourier_trasform_cartesian_prokop, symbols, coords, dictCGFs, number_of_basis, path_hdf5, project_name, orbital)
    
    # Compute the momentum density (an Scalar)
    return partial(fun_density_real, fun_fourier)


def create_alpha_paths(lattice_cte):
    """
    Create all the initial and final paths between gamma alpha and Chi_bb
    """
    def zip_path_coord(initials, finals):
        return concat([list(zip(itertools.repeat(init), fs))
                       for init, fs in zip(initials, finals)])

    initial_alpha_pos = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]

    final_alpha_x = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1)]
    final_alpha_y = [swap(t, 0, 1) for t in final_alpha_x]
    final_alpha_z = [swap(t, 1, 2) for t in final_alpha_y]

    final_positives = [final_alpha_x, final_alpha_y, final_alpha_z]

    positives = zip_path_coord(initial_alpha_pos, final_positives)

    initial_alpha_neg = [mirror_axis(t, i)
                         for i, t in enumerate(initial_alpha_pos)]

    final_negatives = [list(map(lambda xs: mirror_axis(xs, i), fs))
                       for i, fs in enumerate(final_positives)]

    negatives = zip_path_coord(initial_alpha_neg, final_negatives)

    paths = concat([positives, negatives])

    return map_fun(lambda x: x * 2 * pi / lattice_cte, paths)


def create_beta_paths(lattice_cte):
    """
    Create all the initial and final paths between gamma alpha and Chi_bb
    """
    gammas_beta = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1),
                   (1, -1, -1), (-1, 1, -1), (-1, -1, -1)]

    paths = concat([mirror_cube(gamma) for gamma in gammas_beta])

    return map_fun(lambda x: x * 2 * pi / lattice_cte, paths)


def mirror_cube(gamma: Tuple) -> List:
    """
    Find the Chi neighbors of a gamma alpha point
    """
    reflexion_axis = [i for i, x in enumerate(gamma) if x < 0]
    chi_positives = [(1, 0, 1), (1, 1, 0), (0, 1, 1)]

    return [(gamma, apply_reflexion(chi, reflexion_axis))
            for chi in chi_positives]


def apply_reflexion(t: Tuple, xs: List) -> Tuple:
    """
    Apply reflexion operation on the coordinate ``t`` over axis ``xs``
    """
    if not xs:
        return t
    else:
        return reduce(lambda acc, i: mirror_axis(acc, i), xs, t)


def mirror_axis(t: Tuple, i: int) -> Tuple:
    """
    Reflect the coordinate ``i`` in tuple ``t``.
    """
    xs = list(t)
    xs[i] = -t[i]

    return tuple(xs)


def swap(t: Tuple, i: int, j: int) -> Tuple:
    """
    swap entries with indexes i and j in tuple t
    """
    xs = list(t)
    v1, v2 = t[i], t[j]
    xs[i], xs[j] = v2, v1
    return tuple(xs)


def map_fun(f, xs):
    mapTuple = lambda xs: tuple(map(f, xs))
    rs = map(lambda t: (mapTuple(t[0]), mapTuple(t[1])), xs)

    return list(rs)


def point_number_to_compute(size, points) -> Vector:
    """ Compute how many grid points is computed in a given worker """
    res = points % size
    n = points // size

    def fun(rank):
        if res == 0:
            return n
        elif rank < res:
            return n + 1
        else:
            return n

    acc = 0
    xs = np.empty((size, 2), dtype=np.int)
    for i in range(size):
        dim = fun(i)
        xs[i] = acc, acc + dim
        acc += dim

    return xs


def grid_kspace(initial, final, points) -> Matrix:
    """
    make a matrix of dimension Points x 3 containing the coordinates
    in the k-space to be sampled.
    """
    mtx = np.stack([np.linspace(i, f, points) for i, f in zip(initial, final)])

    return np.transpose(mtx)


def read_cmd_line(parser):
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    attributes = ['p', 'hdf5', 'xyz', 'alat', 'basis', 'lower', 'upper']

    return [getattr(args, x) for x in attributes]


def normalize(xs):
    norm = sqrt(np.dot(xs, xs))

    return np.array(xs) / norm


def fun_density_real(function: Callable, k: float) -> float:
    """ Compute the momentum density"""
    return np.absolute(function(k))


if __name__ == "__main__":
    msg = " script -hdf5 <path/to/hdf5> -xyz <path/to/geometry/xyz -alat lattice_cte -b basis_name"

    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('-p', required=True, help='Project name')
    parser.add_argument('-hdf5', required=True, help='path to the HDF5 file')
    parser.add_argument('-xyz', required=True,
                        help='path to molecular gemetry')
    parser.add_argument('-alat', required=True, help='Lattice Constant [Angstroms]',
                        type=float)
    parser.add_argument('-basis', help='Basis Name',
                        default="DZVP-MOLOPT-SR-GTH")
    parser.add_argument('-lower',
                        help='lower orbitals to compute band', type=int,
                        default=19)
    parser.add_argument('-upper',
                        help='upper orbitals to compute band', type=int,
                        default=21)

    main(parser)
