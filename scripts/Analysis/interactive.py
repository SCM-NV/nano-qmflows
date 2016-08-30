
import fnmatch
import numpy as np
import os


def ask_question(q_str, special='None', default=None):
    """
    function that ask the question, and parses the answer to prefered format
    q_str = string containing the question to be asked
    returns string, bool (spec='bool'), int (spec='int') or
    float (spec='float')
    """

    import sys

    while True:
        version = sys.version_info[0]
        question = str(input(q_str)) if version == 3 else str(raw_input(q_str))
        funcs = {'None': [str, {}],
                 'bool': [bool, {'y': True, 'yes': True, 'n': False,
                                 'no': False}],
                 'int': [int, {}], 'float': [float, {}]}

        if not question and default:
            question = str(default)
        if question in funcs[special][1]:
            return funcs[special][1][question]
        elif special is not 'bool':
            try:
                return funcs[special][0](question)
            except ValueError:
                pass
        print("Input not recognised. Please try again.")


def ask_for_states():
    print("""The states are labeled as in PYXAID; the first state being 0.
    You can look them up in the output.""")
    i1 = ask_question("What is the integer representing the first state (int)? ",
                      special='int')
    i2 = ask_question("What is the integer representing the second state (int)? ",
                      special='int')
    return i1, i2


def read_spec_files(i1, i2):
    """
    function that opens all the spectral density files of all the initial
    conditions for states i1 and i2
    and returns
    w - which is a list containing the w-values, which should be the same
    for all initial conditions
    w - type: list of floats
    J - containing lists of the J values for all initial conditions
    J - type: list of lists of floats
    """
    w, j = [], []
    files = os.listdir('.')
    name = 'icond*pair{:d}_{:d}Spectral_density.txt'.format(i1, i2)
    density_files = fnmatch.filter(files, name)
    if density_files:
        for filename in density_files:
            arr = np.loadtxt(filename, usecols=(3, 5))
            arr = np.transpose(arr)
            w.append(arr[0])
            j.append(arr[1])
        return w, j
    else:
        name2 = name[0:4] + '0' + name[6:]
        msg = ('File not found.\nAre you in the out folder? And are '
               'you sure the ints are correct?\n'
               'The program was looking for a file named: \'{}\' in your '
               'current directory.'.format(name2))
        raise FileNotFoundError(msg)
