
from Cython.Distutils import build_ext
from setuptools import Extension, setup


setup(
    name='NonAdiabaticCouling',
    version='0.1.5',
    description='Automation of computations in quantum chemistry',
    license='',
    url='',
    author_email='',
    keywords='chemistry Photochemistry Simulation',
    packages=["nac", "nac.basisSet", "nac.integrals", "nac.schedule",
              "scripts"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'programming language :: python :: 3.5',
        'development status :: 3 - alpha',
        'intended audience :: science/research',
        'topic :: scientific/engineering :: chemistry'
    ],
    install_requires=['cython', 'numpy', 'h5py', 'pymonad'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('obaraSaika', ['nac/integrals/obaraSaika.pyx']),
                 Extension('multipoleObaraSaika', ['nac/integrals/multipoleObaraSaika.pyx'])]
)

