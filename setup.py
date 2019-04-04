from Cython.Distutils import build_ext
from os.path import join
from setuptools import (Extension, find_packages, setup)
import os
import setuptools
import sys


def readme():
    with open('README.rst') as f:
        return f.read()


class get_pybind_include:
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


# Set path to the conda libraries
conda_prefix = os.environ["CONDA_PREFIX"]
if conda_prefix is None:
    raise RuntimeError("No conda module found. A Conda environment is required")

conda_include = join(conda_prefix, 'include')
conda_lib = join(conda_prefix, 'lib')
ext_pybind = Extension(
    'compute_integrals',
    sources=['libint/compute_integrals.cc'],
    include_dirs=[
        # Path to pybind11 headers
        'libint/include',
        conda_include,
        join(conda_include, 'eigen3'),
        join(conda_include, 'python3.6m'),
        get_pybind_include(),
        get_pybind_include(user=True),
        '/usr/include/eigen3'
    ],
    libraries=['hdf5', 'int2'],
    library_dirs=[conda_lib],
    language='c++')


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cc') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fopenmp'):
                opts.append('-fopenmp')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='qmflows-namd',
    version='0.8.0',
    description='Derivative coupling calculation',
    license='Apache-2.0',
    url='https://github.com/SCM-NV/qmflows-namd',
    author=['Felipe Zapata', 'Ivan Infante'],
    author_email='f.zapata@esciencecenter.nl',
    keywords='chemistry Photochemistry Simulation',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'programming language :: python :: 3.7',
        'development status :: 4 - Beta',
        'intended audience :: science/research',
        'topic :: scientific/engineering :: chemistry'
    ],
    install_requires=[
        'numpy', 'h5py', 'noodles==0.3.3', 'pybind11>=2.2.4',
        'pymonad', 'scipy', 'schema', 'pyyaml==5.1',
        'plams@git+https://github.com/SCM-NV/PLAMS@v1.4',
        'qmflows@git+https://github.com/SCM-NV/qmflows@master'
    ],
    dependency_links=[
        "https://github.com/SCM-NV/qmflows/tarball/master#egg=qmflows",
        "git+https://github.com/SCM-NV/PLAMS@v1.4#egg=plams-1.4"],
    cmdclass={'build_ext': BuildExt},
    ext_modules=[ext_pybind],
    extras_require={
        'test': ['coverage', 'pytest>=3.9', 'pytest-cov', 'codacy-coverage']},
    include_package_data=True,
    package_data={
        'nac': ['basis/*.json', 'basis/BASIS*', 'basis/GTH_POTENTIALS']
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
