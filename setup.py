from Cython.Distutils import build_ext
from setuptools import (Extension, find_packages, setup)
from subprocess import (PIPE, Popen)
import os


def check_compiler(compiler: str) -> bool:
    """
    Check that a compiler is available
    """
    cmd = "{} --version".format(compiler)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    rs = p.communicate()
    err = rs[1]
    if err:
        print(compiler, " compiler not avaialable!")
    return True if not err else False


if check_compiler('gcc'):
    flags = ['-fopenmp']


def readme():
    with open('README.rst') as f:
        return f.read()


def check_compiler(cmd: str) -> bool:
    """
    Check that a compiler is available
    """
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    rs = p.communicate()
    err = rs[1]
    if err:
        print(cmd, " compiler not avaialable!")
    return True if not err else False


ext_obara_saika = Extension(
        'multipoleObaraSaika', ['nac/integrals/multipoleObaraSaika.pyx'],
        extra_compile_args=flags,
        extra_link_args=flags)

setup(
    name='qmflows-namd',
    version='0.5.0',
    description='Derivative coupling calculation',
    license='Apache-2.0',
    url='https://github.com/SCM-NV/qmflows-namd',
    author=['Felipe Zapata', 'Ivan Infante'],
    author_email='f.zapata@esciencecenter.nl',
    keywords='chemistry Photochemistry Simulation',
    long_description=readme(),
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'programming language :: python :: 3.6',
        'development status :: 4 - Beta',
        'intended audience :: science/research',
        'topic :: scientific/engineering :: chemistry'
    ],
    install_requires=[
        'cython>=0.29.2', 'numpy', 'h5py', 'noodles==0.3.1',
        'qmflows>=0.3.0', 'pymonad', 'scipy', 'schema', 'pyyaml'],
    dependency_links=[
        "https://github.com/SCM-NV/qmflows/tarball/master#egg=qmflows"],
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_obara_saika],
    extras_require={
        'test': ['coverage', 'pytest>=3.9', 'pytest-cov', 'codacy-coverage']},
    include_package_data=True,
    package_data={
        'nac': ['basisSet/valence_electrons.json']
    },
    scripts=[
        'scripts/cli/run_workflow.py',
        'scripts/distribution/distribute_jobs.py',
        'scripts/hamiltonians/plot_mos_energies.py',
        'scripts/hamiltonians/plot_spectra.py',
        'scripts/pyxaid/plot_average_energy.py',
        'scripts/pyxaid/plot_cooling.py',
        'scripts/pyxaid/plot_spectra_pyxaid.py',
        'scripts/pyxaid/plot_states_pops.py',
        'scripts/qmflows/mergeHDF5.py',
        'scripts/qmflows/plot_dos.py',
        'scripts/qmflows/rdf.py',
        'scripts/qmflows/removeHDF5folders.py',
        'scripts/qmflows/remove_mos_hdf5.py']
)
