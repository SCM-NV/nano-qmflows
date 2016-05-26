__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from itertools import chain
from os.path import join

import os
import yaml
# ==================> Internal modules <==========
from PYXAID import average, lazy, pyxaid_core

# ==============================> Main <==================================


def main():
    pyxaid_args = {'initial_excitation': (102, 108), 'nmin': 90, 'nmax': 122,
                   'homo': 102, 'lumo': 103, 'iconds': [0, 25, 49], 'namdtime': 197}

    #  Environmental Variables
    cwd = os.path.realpath("./")
    work_dir = join(cwd, 'test_pentancene')

    # Path to Hamiltonians PYXAID format
    path_hamiltonians = join(work_dir, 'hamiltonians')

    prefix = join(path_hamiltonians, 'Ham')
    npoints = pyxaid_args['namdtime']
    hams_files = [('{}_{}_im'.format(prefix, i), '{}_{}_re'.format(prefix, i))
                  for i in range(npoints)]

    # path to the PYXAID Templates
    path_template = './data/namd.json'

    output_folder = join(work_dir, 'output_pyxaid')
    params = define_params(hams_files, path_hamiltonians, work_dir, pyxaid_args,
                           path_template, output_folder)

    # Run NAMD
    pyxaid_namd(params)
    
    # Analyze results
    pyxaid_analysis(work_dir, params, output_folder)
    
# ==============================> Tasks <=====================================


def pyxaid_namd(params):
    """Run the actual namd"""
    pyxaid_core.info().version()
    pyxaid_core.namd(params)

    
def pyxaid_analysis(work_dir, params, output_folder, opt=1):
    """
    Define the groups of states for which we want to know the total population
    as a function of time. And run the average function from PYXAID
    """
    states = params['states']
    ms = [[i] for i in range(len(states))]

    res_dir = join(work_dir, 'macro')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    nStates = len(states)
    average.average(params["namdtime"], nStates, params["iconds"], opt, ms,
                    output_folder, res_dir)


def define_params(files_hamiltonians, path_hamiltonians, work_dir,
                  pyxaid_args, path_template, output_folder):
    """
    Using the defaults define in 'nac/templates/namd.json' and the input
    arguments provided by the user to tun nonAdiabatic dinamic simulation
    using the **PYXAID** software.
    Once the parameters have been set the the NAC functions from Pyxaid
    are called and schedule.
    """
    def check_state(f):
        if os.path.exists(f):
            return os.stat(f).st_size != 0
        else:
            return False
    
    # Check if the Hamiltonian files are not empty
    if not all(map(check_state, chain(*files_hamiltonians))):
        msg = 'There is something wrong with the hamiltonian calculations\n'
        raise RuntimeError(msg)

    pyxaid_settings = {}

    # User defined pyxaid setting
    prefix = join(path_hamiltonians, "Ham_")
    pyxaid_settings["Ham_re_prefix"] = prefix
    pyxaid_settings["Ham_im_prefix"] = prefix

    # Folder where the output files are create
    pyxaid_settings["scratch_dir"] = output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Initial condition excitation: I->J
    # This is how I,J are mapped to index of the corresponding basis state
    # see module lazy.py for more details. This is not the most general way
    # of such mapping though.
    i, j = pyxaid_args['initial_excitation']
    homo, lumo = pyxaid_args['homo'], pyxaid_args['lumo']
    nmin, nmax = pyxaid_args['nmin'], pyxaid_args['nmax']
    ex_indx = 1 + (j - lumo) * (homo + 1 - nmin) + (i - nmin)

    # Each entry of the list below is an initial condition. It is also a list
    # but containing only 2 elements - first is the time step at which we start
    # the dynamics, the second is the index of the excited state configuration
    # pyxaid_settings["iconds"] = [[0, ex_indx], [25, ex_indx], [49, ex_indx]]

    iconds = pyxaid_args['iconds']
    pyxaid_settings['iconds'] = [[k, ex_indx] for k in iconds]

    # NAMD time
    pyxaid_settings["namdtime"] = pyxaid_args['namdtime']
    
    # Set active space and the basis states
    pyxaid_settings["active_space"] = range(nmin, nmax + 1)
    
    # Generate basis states
    # ground state
    gs = lazy.ground_state(nmin, homo)
    # single excitations
    ses = lazy.single_excitations(nmin, nmax, homo, 1)

    # Now combine the ground and singly excited states in one list
    # In PYXAID convention, the GS configuration must be the first state in the
    # list of the basis states.
    pyxaid_settings['states'] = [gs] + ses

    # Default pyxaid namd settings

    with open(path_template, 'r') as f:
        params = yaml.safe_load(f)
    # templates = 'templates/namd.json'
    # xs = pkg.resource_string("nac", templates)
    # defaults = yaml.safe_load(xs)

    # finally update the defaults dictionary with the active space information
    params.update(pyxaid_settings)

    return  params

# ============<>===============
if __name__ == "__main__":
    main()
