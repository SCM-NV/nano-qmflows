"""Installation recipe."""
import os
import sys
from os.path import join
from typing import TYPE_CHECKING

import setuptools
import pybind11
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

if TYPE_CHECKING:
    from distutils.ccompiler import CCompiler
    from distutils.errors import CompileError
else:
    CCompiler = setuptools.distutils.ccompiler.CCompiler
    CompileError = setuptools.distutils.errors.CompileError

here = os.path.abspath(os.path.dirname(__file__))

version: "dict[str, str]" = {}
with open(os.path.join(here, 'nanoqm', '_version.py'), 'r', encoding='utf8') as f:
    exec(f.read(), version)


def readme() -> str:
    """Load readme."""
    with open('README.rst', 'r', encoding='utf8') as f:
        return f.read()


def has_flag(compiler: "CCompiler", flagname: str) -> bool:
    """Return a boolean indicating whether a flag name is supported on the specified compiler.

    As of Python 3.6, CCompiler has a `has_flag` method.
    http: // bugs.python.org/issue26689
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cc') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


def cpp_flag(compiler: "CCompiler") -> str:
    """Return the -std=c++[17/14/11] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts: "dict[str, list[str]]" = {
        'msvc': ['/EHsc'],
        'unix': ['-Wall'],
    }
    l_opts: "dict[str, list[str]]" = {
        'msvc': [],
        'unix': ['-Wall'],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14', '-fno-sized-deallocation']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self) -> None:
        """Actual compilation."""
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


def parse_requirements(path: "str | os.PathLike[str]") -> "list[str]":
    """Parse a ``requirements.txt`` file and strip all empty and commented lines."""
    ret = []
    with open(path, "r", encoding="utf8") as f:
        for i in f:
            j = i.split("#", 1)[0].strip().rstrip()
            if j:
                ret.append(j)
    return ret


# Set path to the conda libraries
conda_prefix = os.environ["CONDA_PREFIX"]
if conda_prefix is None:
    raise RuntimeError(
        "No conda module found. A Conda environment is required")


conda_include = join(conda_prefix, 'include')
conda_lib = join(conda_prefix, 'lib')
ext_pybind = Extension(
    'nanoqm.compute_integrals',
    sources=['libint/compute_integrals.cc'],
    include_dirs=[
        # Path to pybind11 headers
        'libint/include',
        conda_include,
        join(conda_include, 'eigen3'),
        '/usr/include/eigen3'
        pybind11.get_include(),
    ],
    libraries=['hdf5', 'int2'],
    library_dirs=[conda_lib],
    language='c++',
)

setup(
    name='nano-qmflows',
    version=version['__version__'],
    description='Derivative coupling calculation',
    license='Apache-2.0',
    url='https://github.com/SCM-NV/nano-qmflows',
    author='Felipe Zapata & Ivan Infante',
    author_email='f.zapata@esciencecenter.nl',
    keywords='chemistry Photochemistry Simulation',
    long_description=readme() + '\n\n',
    long_description_content_type='text/x-rst',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Typing :: Typed',
    ],
    install_requires=parse_requirements("install_requirements.txt"),
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.7',
    ext_modules=[ext_pybind],
    extras_require={
        'test': parse_requirements("test_requirements.txt"),
        'doc': parse_requirements("doc_requirements.txt"),
    },
    include_package_data=True,
    package_data={
        'nanoqm': ['basis/*.json', 'basis/BASIS*', 'basis/GTH_POTENTIALS', 'py.typed', '*.pyi']
    },
    entry_points={
        'console_scripts': [
            'run_workflow.py=nanoqm.workflows.run_workflow:main',
            'distribute_jobs.py=nanoqm.workflows.distribute_jobs:main'
        ]
    },
    scripts=[
        'scripts/convert_legacy_hdf5.py',
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
        'scripts/qmflows/remove_mos_hdf5.py',
        'scripts/qmflows/convolution.py'
    ]
)
