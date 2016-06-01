__author__ = "Felipe Zapata"


import numpy as np

# ==================> Internal modules <==========
from noodles import gather
from qmworks import run
from qmworks.utils import concatMap


# =====================================<>======================================




def calculateGrid(timeCoeffs, mo, center=(0, 0, 0), nPoints=(80, 80, 80),
                  nmin=None, nmax=None):
    """
    """


def read_cube_file(path_to_cube, numat):
    """
    Read a cube files and return the array together with the grid
    parameters.

    :param path_to_cube: path to the cube file to read
    :type path_to_cube: String
    """
    with open(path_to_cube, 'r') as f:
        head = [next(f) for i in range(6)]
    # There are 2 initial lines cotaining comments.
    # Then one line cotaining the grid center and finally
    # 3 lines with the number of voxels and the axis vector. 
    num_lines = numat + 6  # 
    np.loadtxt(path_to_cube, skiprows=num_lines)


    def write_cube_file(center, coordinates, grid_spec, arr):
    """
    Write some density using the cubefile format
    :param center: Center of the grid
    :type center:
    """
    def formatCols(xs, cols=4):
        if cols == 4:
            string = '{} {:10.6f} {:10.6f} {:10.6f}\n'
        elif cols == 5:
            string = '{} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n'
        else:
            msg = "There is not format for that number of columns "
            raise NotImplementedError(msg)

        return string.format(*xs)
    
    def printAtom(xs):
        """
        Atoms row have the following format:
        6    0.000000  -11.020792    0.648172    0.001778
        where the first colums specific the atomic number, the
        second apparently does not have a clear meaning and
        the last three are the cartesian coordinates.
        """
        rs = [xs[0]] + [0] + xs[1:]

        return formatCols(rs, cols=5)

    inp = 'density\ndensity\n'
    inp += formatCols([numat] + center)
    inp += formatCols([grid_spec[0, 0], grid_spec[0, 1], 0, 0])
    inp += formatCols([grid_spec[0, 0], 0, grid_spec[1, 1], 0])
    inp += formatCols([grid_spec[0, 0], 0, 0, grid_spec[2, 1]])
    inp += concatMap(printAtom, coordinates)
    inp += 

    return inp
    

    
