
import sys
sys.path.append('/home/prokop/git_SW/nonAdiabaticCoupling')

data_dir = "/home/prokop/Desktop/kscan_qmworks/Si68-H"

from functools import (partial, reduce)
from math import (pi, sqrt)
from multiprocessing import Pool
from nac.integrals.fourierTransform import calculate_fourier_trasform_cartesian, calculate_fourier_trasform_cartesian_prokop, get_fourier_basis
from nac.schedule.components        import create_dict_CGFs
from os.path import join
from qmworks.parsers.xyzParser import readXYZ
from qmworks.utils import concat

from nac import retrieve_hdf5_data

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
        if (iE>0) and (iE<nE):
            bins[iE,:] = np.maximum( bins[iE,:], kdata[i,:] )
    extent=(0,1,Emin,Emax)
    return bins, extent











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
    atoms = readXYZ(path_xyz)
    symbols = np.array([at.symbol for at in atoms])
    coords_angstrom = np.concatenate([at.xyz for at in atoms])
    angstroms_to_au = 1.889725989
    coords = angstroms_to_au * coords_angstrom
    lattice_cte = lattice_cte * angstroms_to_au

    # Dictionary containing as key the atomic symbols and as values the set of CGFs
    dictCGFs = create_dict_CGFs(path_hdf5, basis_name, atoms)
    count_cgfs = np.vectorize(lambda s: len(dictCGFs[s]))
    number_of_basis = np.sum(np.apply_along_axis(count_cgfs, 0, symbols))

    # K-space grid to calculate the fuzzy band
    nPoints = 20
    # grid_k_vectors = grid_kspace(initial, final, nPoints)
    map_grid_kspace = lambda ps: [grid_kspace(i, f, nPoints) for i, f in ps]

    # Grid
    grids_alpha = map_grid_kspace(create_alpha_paths(lattice_cte))
    grids_beta = map_grid_kspace(create_beta_paths(lattice_cte))

    # Apply the whole fourier transform to the subset of the grid
    # correspoding to each process
    momentum_density = partial(compute_momentum_density, project_name, symbols,
                               coords, dictCGFs, number_of_basis, path_hdf5)

    orbitals = list(range(lower, upper + 1))
    dim_x    = len(orbitals)
    result   = np.empty((dim_x, nPoints))
    '''
    # original Fillipe Zappata
    with Pool(processes=1) as p:
        for i, orb in enumerate(orbitals):
            print("Orbital: ", orb)
            density = momentum_density(orb)
            alphas = [normalize(p.map(density, grid_k))
                      for grid_k in grids_alpha]
            betas = [normalize(p.map(density, grid_k))
                     for grid_k in grids_beta]
            print( alphas)
            rs_alphas = normalize(np.stack(alphas).sum(axis=0))
            rs_betas  = normalize(np.stack(betas).sum(axis=0))
            rss       = normalize(rs_alphas + rs_betas)
            result[i] = rss
            np.save('alphas', rs_alphas)
            np.save('betas', rs_betas)
            print("Orb: ", orb)
            print(rss)
    '''
    '''
    # original Fillipe Zappata - but removed renormalization
    # [ 0.1010494   0.12795925  0.11550439  0.09820901  0.14555559  0.14273843
    # 0.08865795  0.07344947  0.06574112  0.05107196  0.06109847  0.07562495
    # 0.12683827  0.17763785  0.18248881  0.29108211  0.30235151  0.49110912
    # 0.89813029  1.08569162]
    with Pool() as p:
        for i, orb in enumerate(orbitals):
            print("Orbital: ", orb)
            density = momentum_density(orb)
            alphas = [ p.map(density, grid_k) for grid_k in grids_alpha ]
            betas  = [ p.map(density, grid_k) for grid_k in grids_beta  ]
            print( alphas)
            rs_alphas = np.stack(alphas).sum(axis=0)
            rs_betas  = np.stack(betas).sum(axis=0) 
            rss       = rs_alphas + rs_betas
            result[i] = rss
            np.save('alphas', rs_alphas)
            np.save('betas' , rs_betas)
            print("Orb: ", orb)
            print(rss)
    '''
    '''
    # Prokop Hapala V1 - reproduce fillipe's result but normalization should not be here
    # result it produces for Si68H orb 19. 
    # with normalization
    #[ 0.06289826  0.07835282  0.06883761  0.06073309  0.09062537  0.08889299
    #0.05469662  0.04578841  0.04325191  0.03772494  0.04611629  0.05368498
    #0.08025384  0.1093876   0.11306399  0.18438708  0.18586155  0.28444182
    #0.55230157  0.68285762]
    # without normalization
    #[ 0.05664861  0.0767676   0.06960464  0.05724898  0.08921345  0.08777105
    #0.05197325  0.04430297  0.04071755  0.03115768  0.03770548  0.04530169
    #0.07558461  0.10453225  0.1040781   0.17481291  0.17713627  0.29428565
    #0.56023894  0.68298257]
    for i, orb in enumerate(orbitals):
        print("Orbital: ", orb)
        density   = momentum_density(orb)
        alphas    = np.array([ density( grid_k) for grid_k in grids_alpha ])
        betas     = np.array([ density( grid_k) for grid_k in grids_beta  ])
        #alphas    = np.array([ normalize(density( grid_k)) for grid_k in grids_alpha ])
        #betas     = np.array([ normalize(density( grid_k)) for grid_k in grids_beta  ])
        #rs_alphas = normalize(np.stack(alphas).sum(axis=0))
        #rs_betas  = normalize(np.stack(betas).sum(axis=0))
        #rss       = normalize(rs_alphas + rs_betas)
        
        result[i] = rss
        #np.save('alphas', rs_alphas)
        #np.save('betas', rs_betas)
        print("Orb: ", orb)
        print(rss)      
    '''
    
    # Prokop Hapala V2 - not yet working - probably due to normalization
    grids_alpha = np.array(grids_alpha)
    grids_beta  = np.array(grids_beta)
    nalpha      = grids_alpha.shape[0] * grids_alpha.shape[1]
    nbeta       = grids_beta .shape[0] * grids_beta .shape[1]
    kpoints     = np.concatenate( (grids_alpha.reshape(-1,3),grids_beta.reshape(-1,3)) ) 
    print ( "shapes:", grids_alpha.shape, grids_beta.shape, kpoints.shape, nalpha, nbeta )
    #print ( kpoints )
    #exit()
    print ("building basiset fourier dictionary ... ")
    chikdic = get_fourier_basis( symbols, dictCGFs, kpoints )
    print ("...fourier basis DONE !")
    for i, orb in enumerate(orbitals):
        print("Orbital: ", orb)
        orbK      = calculate_fourier_trasform_cartesian_prokop( symbols, coords, dictCGFs, number_of_basis, path_hdf5, project_name, orb, kpoints, chikdic=chikdic )
        orbK      = np.absolute( orbK )
        alphas    = orbK[:nalpha].reshape(grids_alpha.shape[0],grids_alpha.shape[1])
        betas     = orbK[nalpha:].reshape(grids_beta.shape[0],grids_beta.shape[1])
        #rs_alphas = normalize(np.stack(alphas).sum(axis=0))
        #rs_betas  = normalize(np.stack(betas).sum(axis=0))
        #rss       = normalize(rs_alphas + rs_betas)
        rs_alphas = np.stack(alphas).sum(axis=0)
        rs_betas  = np.stack(betas).sum(axis=0)
        rss       = rs_alphas + rs_betas
        result[i] = rss
        #np.save('alphas', rs_alphas)
        #np.save('betas', rs_betas)
        #print("Orb: ", orb)
        #print(rss)  
        #plt.plot( range(len(rss)),rss )
    
    print (result[19-1])
    
    path_energy = 'Si68-H/point_0/cp2k/mo/eigenvalues'
    Es = retrieve_hdf5_data(join(data_dir,'quantum.hdf5'), [path_energy])[0]
    #print( "Es = ", Es )
    Es = [ Es[i]*27.2114 for i in orbitals]
    #print( "Es = ", Es )
    
    bins, extent = projectionsToBins( result, Es, Emin=-6.0, Emax=0.0, dE=0.02 )
    
    plt.figure(figsize=(3,8))
    plt.imshow( np.log10(bins), interpolation='nearest', origin='image', extent=extent, cmap='jet' )
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
