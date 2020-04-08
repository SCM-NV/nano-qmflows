Using coordination_ldos.py
--------------------------

The script prints local PDOS projected on subsets of atoms given through lists.
These lists are obtained using the nano-CAT module ``nanoCAT.recipes.coordination_number`` (see the relative documentation_)
that returns a nested dictionary

``{'Cd': {4: [0, 1, 2, 3, 4, ...], ...}, ...}``

with atomic symbol (*e.g.* ``'Cd'``) and coordination number (*e.g.* ``4``) as keys.


You thus have to install the nano-CAT package in your conda environment according to the installation instructions, reported here_.

.. _documentation: https://cat.readthedocs.io/en/latest/12_5_recipes.html
.. _here: https://github.com/nlesc-nano/nano-CAT
