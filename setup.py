
from Cython.Distutils import build_ext
from setuptools import Extension, setup
import numpy as np
import os
import shutil

if shutil.which('icc') is not None:
    os.environ['CC'] = 'icc'
    os.environ['LDSHARED'] = 'icc -shared'

setup(
    name='qmflows-namd',
    version='0.3.0',
    description='Derivative coupling calculation',
    license='Apache-2.0',
    url='https://github.com/SCM-NV/qmflows-namd',
    author=['Felipe Zapata', 'Ivan Infante'],
    author_email='tifonzafel_gmail.com',
    keywords='chemistry Photochemistry Simulation',
    packages=[
        "nac", "nac.analysis", "nac.basisSet", "nac.integrals", "nac.schedule",
        "nac.workflows"],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'programming language :: python :: 3.5',
        'development status :: 4 - Beta',
        'intended audience :: science/research',
        'topic :: scientific/engineering :: chemistry'
    ],
    install_requires=[
        'cython', 'numpy', 'h5py', 'noodles==0.2.4', 'numba', 'qmflows', 'pymonad', 'scipy'],
    dependency_links=[
            "https://github.com/SCM-NV/qmflows/tarball/master#egg=qmflows"],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(
        'multipoleObaraSaika', ['nac/integrals/multipoleObaraSaika.pyx'])],
    include_dirs=[np.get_include()],
    extras_require={'test': ['coverage', 'pytest', 'pytest-cov']},
    scripts=[
        'scripts/hamiltonians/plot_mos_energies.py',
        'scripts/hamiltonians/plot_spectra.py',
        'scripts/pyxaid/plot_average_energy.py',
        'scripts/pyxaid/plot_spectra_pyxaid.py',
        'scripts/pyxaid/plot_states_pops.py',
        'scripts/qmflows/mergeHDF5.py',
        'scripts/qmflows/removeHDF5folders.py',
        'scripts/qmflows/remove_mos_hdf5.py',
        'scripts/pyxaid/plot_cooling.py',
        'scripts/distribution/distribute_jobs.py',
        'scripts/distribution/merge_job.py',
        'scripts/qmflows/plot_dos.py']
)
