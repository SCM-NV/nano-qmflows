
from os.path import join
import h5py


def main():
    """
    """
    ks = ['point_{}'.format(i) for i in range(781)]
    with h5py.File("copy.hdf5", 'r') as f5, h5py.File("quantum.hdf5", 'r') as f6:
        for k in ks:
            path = join("Cd68Se55", k)
            if path not in f6:
                f5.copy(path, f6["Cd68Se55"])


if __name__ == '__main__':
    main()
