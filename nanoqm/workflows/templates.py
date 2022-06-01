"""Defaults to call CP2K.

Index
-----
.. currentmodule:: nanoqm.workflows.templates
.. autosummary::
  create_settings_from_template

"""

from __future__ import annotations

__all__ = ["create_settings_from_template"]

import os
import yaml
from scm.plams import Molecule

from qmflows import Settings
from qmflows.type_hints import PathLike
from nanoqm.common import UniqueSafeLoader, valence_electrons, aux_fit
from typing import Any, Dict, Iterable, FrozenSet


def generate_auxiliar_basis(
        sett: Settings, auxiliar_basis: str, quality: str) -> Settings:
    """Generate the `auxiliar_basis` for all the atoms in the `sett`.

    Use the`quality` of the auxiliar basis provided by the user.
    """
    quality_to_number = {"low": 0, "medium": 1,
                         "good": 2, "verygood": 3, "excellent": 4}
    kind = sett.cp2k.force_eval.subsys.kind
    for atom in kind.keys():
        index = quality_to_number[quality.lower()]
        cfit = aux_fit[atom][index]
        kind[atom]["basis_set"].append(f"AUX_FIT CFIT{cfit}")

    return sett


#: Settings for a PBE calculation to compute a guess wave function
cp2k_guess = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy
  force_eval:
    dft:
      scf:
        eps_scf: 1e-6
        added_mos: 0
        scf_guess: "restart"
        ot:
          minimizer: "DIIS"
          n_diis: 7
          preconditioner: "FULL_SINGLE_INVERSE"
""", Loader=UniqueSafeLoader))

#: Settings for a PBE calculation to compute the Molecular orbitals
cp2k_main = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy

  force_eval:
    dft:
      scf:
        eps_scf: 1e-06
        max_scf: 200
        scf_guess: "restart"
""", Loader=UniqueSafeLoader))


#: Settings for a PBE calculation to compute a guess wave function
cp2k_pbe_guess = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy
  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional pbe: {}
      scf:
        eps_scf: 1e-6
        added_mos: 0
        scf_guess: "restart"
        ot:
          minimizer: "DIIS"
          n_diis: 7
          preconditioner: "FULL_SINGLE_INVERSE"
""", Loader=UniqueSafeLoader))

#: Settings for a PBE calculation to compute the Molecular orbitals
cp2k_pbe_main = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy

  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional pbe: {}
      scf:
        eps_scf: 1e-06
        max_scf: 200
        scf_guess: "restart"
""", Loader=UniqueSafeLoader))

#: Settings for a R2SCAN calculation to compute a guess wave function
cp2k_scan_guess = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy
  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional:
          mgga_x_r2scan:
            scale: 1.0
          mgga_c_r2scan:
            scale: 1.0
        xc_grid:
          xc_deriv: "spline3"
          xc_smooth_rho: "None"
      scf:
        eps_scf: 1e-6
        added_mos: 0
        scf_guess: "restart"
        ot:
          minimizer: "DIIS"
          n_diis: 7
          preconditioner: "FULL_SINGLE_INVERSE"
""", Loader=UniqueSafeLoader))

#: Settings for a R2SCAN calculation to compute the Molecular orbitals
cp2k_scan_main = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy

  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional:
          mgga_x_r2scan:
            scale: 1.0
          mgga_c_r2scan:
            scale: 1.0
        xc_grid:
          xc_deriv: "spline3"
          xc_smooth_rho: "None"
      scf:
        eps_scf: 1e-06
        max_scf: 200
        scf_guess: "restart"
""", Loader=UniqueSafeLoader))

#: Settings for a PBE0 calculation to compute a guess wave function
cp2k_pbe0_guess = Settings(yaml.load("""
cp2k:
   global:
     run_type:
       energy

   force_eval:
     subsys:
       cell:
         periodic: "None"
     dft:
       auxiliary_density_matrix_method:
         method: "basis_projection"
         admm_purification_method: "none"
       qs:
         method: "gpw"
         eps_pgf_orb: 1E-8
       xc:
        xc_functional:
          pbe:
            scale_x: 0.75
            scale_c: 1.00
        hf:
          fraction: 0.25
          screening:
            eps_schwarz: 1.0E-6
            screen_on_initial_p: "True"
          interaction_potential:
            potential_type: "truncated"
            cutoff_radius: 2.5
          memory:
            max_memory: 5000
            eps_storage_scaling: "0.1"
       scf:
          eps_scf: 1e-6
          added_mos: 0
          scf_guess: "restart"
          ot:
            minimizer: "DIIS"
            n_diis: 7
            preconditioner: "FULL_SINGLE_INVERSE"

""", Loader=UniqueSafeLoader))

#: Settings for a PBE0 calculation to compute the Molecular orbitals
cp2k_pbe0_main = Settings(yaml.load("""
cp2k:
   global:
     run_type:
       energy

   force_eval:
     subsys:
       cell:
         periodic: "None"
     dft:
       auxiliary_density_matrix_method:
         method: "basis_projection"
         admm_purification_method: "none"
       qs:
         method: "gpw"
         eps_pgf_orb: "1.0E-8"
       xc:
         xc_functional:
           pbe:
             scale_x: "0.75"
             scale_c: "1.00"
         hf:
           fraction: "0.25"
           screening:
             eps_schwarz: 1.0E-6
             screen_on_initial_p: "True"
           interaction_potential:
             potential_type: "truncated"
             cutoff_radius: 2.5
           memory:
             max_memory: "5000"
             eps_storage_scaling: "0.1"
       scf:
          eps_scf: 1e-06
          max_scf: 200
          scf_guess: "restart"
""", Loader=UniqueSafeLoader))

#: Settings for a HSE06 calculation to compute a guess wave function
cp2k_hse06_guess = Settings(yaml.load("""
cp2k:
   global:
     run_type:
       energy

   force_eval:
     subsys:
       cell:
         periodic: "None"
     dft:
       auxiliary_density_matrix_method:
         method: "basis_projection"
         admm_purification_method: "none"
       qs:
         method: "gpw"
         eps_pgf_orb: 1E-8
       xc:
        xc_functional:
          pbe:
            scale_x: 0.00
            scale_c: 1.00
          xwpbe:
            scale_x: -0.25
            scale_x0: 1.00
            omega: 0.11
        hf:
          fraction: 0.25
          screening:
            eps_schwarz: 1.0E-6
            screen_on_initial_p: "True"
          interaction_potential:
            potential_type: "shortrange"
            omega: 0.11
          memory:
            max_memory: 5000
            eps_storage_scaling: "0.1"
       scf:
          eps_scf: 1e-6
          added_mos: 0
          scf_guess: "restart"
          ot:
            minimizer: "DIIS"
            n_diis: 7
            preconditioner: "FULL_SINGLE_INVERSE"

""", Loader=UniqueSafeLoader))

#: Settings for a HSE06 calculation to compute the Molecular orbitals
cp2k_hse06_main = Settings(yaml.load("""
cp2k:
   global:
     run_type:
       energy

   force_eval:
     subsys:
       cell:
         periodic: "None"
     dft:
       auxiliary_density_matrix_method:
         method: "basis_projection"
         admm_purification_method: "none"
       qs:
         method: "gpw"
         eps_pgf_orb: "1.0E-8"
       xc:
        xc_functional:
          pbe:
            scale_x: 0.00
            scale_c: 1.00
          xwpbe:
            scale_x: -0.25
            scale_x0: 1.00
            omega: 0.11
        hf:
          fraction: 0.25
          screening:
            eps_schwarz: 1.0E-6
            screen_on_initial_p: "True"
          interaction_potential:
            potential_type: "shortrange"
            omega: 0.11
          memory:
            max_memory: 5000
            eps_storage_scaling: "0.1"
       scf:
          eps_scf: 1e-6
          max_scf: 200
          scf_guess: "restart"
""", Loader=UniqueSafeLoader))

#: Settings for a B3LYP calculation to compute a guess wave function
cp2k_b3lyp_guess = Settings(yaml.load("""
cp2k:
   global:
      run_type:
         energy

   force_eval:
      subsys:
         cell:
           periodic: "None"
      dft:
         xc:
           xc_functional b3lyp: {}
         scf:
           eps_scf: 1e-6
           added_mos: 0
           scf_guess: "restart"
           ot:
             minimizer: "DIIS"
             n_diis: 7
             preconditioner: "FULL_SINGLE_INVERSE"
""", Loader=UniqueSafeLoader))

#: Settings for a B3LYP calculation to compute the Molecular orbitals
cp2k_b3lyp_main = Settings(yaml.load("""
cp2k:
  global:
    run_type:
      energy

  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional b3lyp: {}
      scf:
        eps_scf: 1e-06
        max_scf: 200
        scf_guess: "restart"
""", Loader=UniqueSafeLoader))


#: Settings to add the CP2K kinds for each atom
kinds_template = Settings(yaml.load("""
cp2k:
  force_eval:
    subsys:
      kind:
        C:
          basis_set: DZVP-MOLOPT-SR-GTH-q4
          potential: GTH-PBE-q4
""", Loader=UniqueSafeLoader))


def generate_kinds(elements: Iterable[str], basis: str, potential: str) -> Settings:
    """Generate the kind section for cp2k basis."""
    s = Settings()
    subsys = s.cp2k.force_eval.subsys
    for e in elements:
        q = valence_electrons[e]
        subsys.kind[e]['basis_set'] = [f"{basis}-q{q}"]
        subsys.kind[e]['potential'] = f"{potential}-q{q}"

    return s


#: available templates
templates_dict = {
    "guess": cp2k_guess, "main": cp2k_main,
    "pbe_guess": cp2k_pbe_guess, "pbe_main": cp2k_pbe_main,
    "scan_guess": cp2k_scan_guess, "scan_main": cp2k_scan_main,
    "pbe0_guess": cp2k_pbe0_guess, "pbe0_main": cp2k_pbe0_main,
    "hse06_guess": cp2k_hse06_guess, "hse06_main": cp2k_hse06_main,
    "b3lyp_guess": cp2k_b3lyp_guess, "b3lyp_main": cp2k_b3lyp_main}


def create_settings_from_template(
    general: Dict[str, Any],
    template_name: str,
    path_traj_xyz: str | os.PathLike[str],
) -> Settings:
    """Create a job Settings using the name provided by the user."""
    setts = templates_dict[template_name]
    elements = read_unique_atomic_labels(path_traj_xyz)

    kinds = generate_kinds(elements, general['basis'], general['potential'])

    if 'pbe0' in template_name:
        s = Settings()
        return generate_auxiliar_basis(setts + s + kinds, general['basis'], general['aux_fit'])
    elif 'hse06' in template_name:
        return generate_auxiliar_basis(setts + kinds, general['basis'], general['aux_fit'])
    else:
        return setts + kinds


def read_unique_atomic_labels(path_traj_xyz: str | os.PathLike[str]) -> FrozenSet[str]:
    """Return the unique atomic labels."""
    mol = Molecule(path_traj_xyz, 'xyz')

    return frozenset(at.symbol for at in mol.atoms)
