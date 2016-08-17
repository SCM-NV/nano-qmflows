
from fnmatch import fnmatch
from os.path import join
from subprocess import PIPE, Popen

import argparse
import h5py
import os

# ====================================<>=======================================
msg = " script -p ProjectName -f5 <path/to/hdf5> -r <path/to/results/files"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-p', required=True,
                    help='Project Name')
parser.add_argument('-f5', required=True,
                    help='Path to the HDF5')
parser.add_argument('-r', required=True,
                    help='Path to the calculations Output')


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    project = args.p
    f5 = args.f5
    res = args.r

    return project, f5, res

# ====================================<>=======================================


def readNonConvergedData(path):
    """
    Search for nonconverged calculation inside path
    :parameter path: path to the output files that failed.
    :returns: paths to the failed jobs.
    """
    cmd = "grep -ir 'NOT converged' --include \*.out {}".format(path)

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    xs, errors = p.communicate()
    print(xs, errors)
    p.terminate()

    lines = xs.splitlines()
    # Take the path and drop the rest of the line
    outputs = map(lambda r: r.split(b':')[0], lines)
    # Remove last entry in path
    dirs = map(lambda x: join(*x.split(b'/')[:-1]), outputs)

    return map(lambda z: z.decode(), dirs)


def readFailedPoints(path):
    """
    Gets the name of the point that file from a folder
    """
    ls = os.listdir(path)
    xs = filter(lambda x: fnmatch(x, "*.wfn"), ls)
    head = list(xs)[0]
    
    return head.split('-')[0]


def deletePointsfromHDF5(pathHDF5, points, projectName):
    """
    Delete Wrong Data from HDF5
    """
    with h5py.File(pathHDF5, 'r+') as f5:
        for p in points:
            node = join(projectName, p)
            if node in f5:
                print("deleting the node at: ", node)
                del f5[node]
        

def main():
    projectName, pathHDF5, pathOutput = read_cmd_line()
    xs = map(readFailedPoints, readNonConvergedData(pathOutput))
    failPoints = list(set(xs)) # unique
    deletePointsfromHDF5(pathHDF5, failPoints, projectName)
    

# ====================================<>=======================================


if __name__ == "__main__":
    main()
