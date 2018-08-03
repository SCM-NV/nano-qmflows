
from mpi4py import MPI
from nac.integrals.multipoleIntegrals import compute_block_triang_indices
import argparse
import numpy as np
import dill

msg = " call_mpi -f <Path/to/serialized/function> -n number_of_orbitals -o <output/array>"

parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-f', required=True,
                    help='Path to function to function to execute the mpi')
parser.add_argument('-n', required=True,
                    help='number of orbitals', type=int)
parser.add_argument('-o', required=True,
                    help='Name of the output array')


def read_cmd_line():
    """
    Parse Command line options.
    """
    args = parser.parse_args()
    nOrbs = args.n
    serialized_file = args.f
    output_file = args.o

    with open(serialized_file, 'rb') as f:
        fun = dill.load(f)

    return fun, nOrbs, output_file


def main():
    fun, nOrbs, output_file = read_cmd_line()

    # start MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # compute the matrix to scatter among the workers
    block_triang_indices = compute_block_triang_indices(nOrbs, size)

    # compute the array
    result = fun(block_triang_indices[rank])

    # Gather the results
    recvbuf = None
    if rank == 0:
        shape_result = (nOrbs ** 2 + nOrbs) // 2
        recvbuf = np.empty(shape_result, dtype=np.float64)
    comm.Gatherv(result, recvbuf, root=0)

    if rank == 0:
        np.save(output_file, recvbuf)


if __name__ == "__main__":
    main()
