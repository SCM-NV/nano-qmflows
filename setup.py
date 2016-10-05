
from Cython.Distutils import build_ext
from setuptools import Extension, setup


setup(
    name='NonAdiabaticCouling',
    version='0.1.6',
    description='Automation of computations in quantum chemistry',
    license='MIT',
    url='https://github.com/felipeZ/nonAdiabaticCoupling',
    author='Felipe Zapata',
    author_email='tifonzafel_gmail.com',
    keywords='chemistry Photochemistry Simulation',
    packages=["nac", "nac.basisSet", "nac.integrals", "nac.schedule",
              "nac.workflows"],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'programming language :: python :: 3.5',
        'development status :: 4 - Beta',
        'intended audience :: science/research',
        'topic :: scientific/engineering :: chemistry'
    ],
    install_requires=['cython', 'numpy', 'h5py', 'noodles', 'qmworks', 'pymonad'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('multipoleObaraSaika', ['nac/integrals/multipoleObaraSaika.pyx'])],
    extras_require={'test': ['nose', 'coverage']}
)
