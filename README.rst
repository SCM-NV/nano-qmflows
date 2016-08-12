
.. image:: https://travis-ci.org/felipeZ/nonAdiabaticCoupling.svg?branch=master
    :target: https://travis-ci.org/felipeZ/nonAdiabaticCoupling
.. image:: https://img.shields.io/github/license/felipeZ/nonAdiabaticCoupling.svg?maxAge=2592000   :target: https://github.com/felipeZ/nonAdiabaticCoupling/blob/master/LICENSE
.. image:: https://img.shields.io/github/release/felipeZ/nonAdiabaticCoupling.svg?maxAge=2592000   :target: https://github.com/felipeZ/nonAdiabaticCoupling/releases/tag/0.1.6
====================
NonAdiabaticCoupling
====================

Package to calculate several properties related to the Nonadiabatic behaviour of a molecular system.

 
Installation
============

- type the following command in ther terminal: ::
    ``git clone git@github.com:felipeZ/nonAdiabaticCoupling.git``

- Then move to the new folder called *nonAdibatic* and type: ::
    ``pip install .`` 

Overview
========
This library contains both a library and a set of scripts to carry out a numerical approximation
to the *nonadibatic coupling". The Library contains the numerical routines written in cython_ 
and numpy_ that are the core of the library. While the scripts are set of workflows to compute different properties using different approximations that can be tuned by the user.

.. _cython: http://cython.org
.. _numpy: http://www.numpy.org

Worflow to calculate Hamiltonians for nonadiabatic molecular simulations
************************************************************************
The figure represents schematically a Worflow to compute the **Hamiltonians** that described the behavior and coupling between the excited state of a molecular system. These **Hamiltonians** are used by thy PYXAID_ simulation package to carry out nonadiabatic molecular dynamics.

.. image:: docs/images/nac_worflow.png

.. _PYXAID: https://www.acsu.buffalo.edu/~alexeyak/pyxaid/overview.html
