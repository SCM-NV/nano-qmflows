from Cython.Distutils import build_ext
from setuptools import (Extension, find_packages, setup)
import os
import setuptools
import shutil
import sys

if shutil.which('icc') is not None:
    os.environ['CC'] = 'icc'
    os.environ['LDSHARED'] = 'icc -shared'


def readme():
    with open('README.rst') as f:
        return f.read()


ext_obara_saika = Extension(
        'multipoleObaraSaika', ['nac/integrals/multipoleObaraSaika.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])


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


ext_pybind = Extension(
    'compute_integrals',
    sources=['libint/compute_integrals.cc'],
    include_dirs=[
        # Path to pybind11 headers
        '/home/felipe/modules/libint/include',
        '/home/felipe/modules/libint/include/libint2',
        get_pybind_include(),
        get_pybind_include(user=True),
        '/usr/include/eigen3'
    ],
    libraries=['int2'],
    library_dirs=['/home/felipe/miniconda3/envs/namd/lib'],
    language='c++'

)


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
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


def call_setup(ext, builder):
    """
    Build `ext` uising `builder`.
    """
    setup(
        name='qmflows-namd',
        version='0.6.0',
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
            'cython>=0.29.2', 'numpy', 'h5py', 'noodles==0.3.1', 'pybind11>=2.2.4',
            'qmflows>=0.3.0', 'pymonad', 'scipy', 'schema', 'pyyaml'],
        dependency_links=[
            "https://github.com/SCM-NV/qmflows/tarball/master#egg=qmflows"],
        cmdclass={'build_ext': builder},
        ext_modules=[ext],
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


call_setup(ext_pybind, BuildExt)

# call_setup(ext_obara_saika, build_ext)
