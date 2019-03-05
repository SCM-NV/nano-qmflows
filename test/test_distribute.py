from subprocess import (PIPE, Popen)
import fnmatch
import shutil
import os


def test_distribute(tmp_path):
    """
    Check that the scripts to compute a trajectory are generated correctly
    """
    cmd1 = "distribute_jobs.py -i test/test_files/input_test_distribute_derivative_couplings.yml"
    call_distribute(tmp_path, cmd1)


def call_distribute(tmp_path, cmd):
    """
    Execute the distribute script and check that if finish succesfully.
    """
    try:
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = p.communicate()
        if err:
            raise RuntimeError(err)
        check_scripts()
    finally:
        remove_chunk_folder()


def check_scripts():
    """
    Check that the distribution scripts were created correctly
    """
    paths = fnmatch.filter(os.listdir('.'), "chunk*")
    assert len(paths) == 5

    for p in paths:
        xs = fnmatch.filter(os.listdir(p), "*")
        assert len(xs) == 3


def remove_chunk_folder():
    """ Remove resulting scripts """
    for path in fnmatch.filter(os.listdir('.'), "chunk*"):
        shutil.rmtree(path)
