
__all__ = ["create_settings_from_template"]

from scm.plams import Molecule
from qmflows.settings import Settings
import json
import pkg_resources as pkg
import yaml


path_valence_electrons = pkg.resource_filename("nac", "basisSet/valence_electrons.json")
with open(path_valence_electrons, 'r') as f:
    valence_electrons = json.load(f)


def generate_auxiliar_basis(sett: Settings, auxiliar_basis: str) -> Settings:
    """
    Generate the `auxiliar_basis` for all the atoms in the `sett`
    """
    kind = sett.cp2k.force_eval.subsys.kind
    for atom in kind.keys():
        kind[atom]["BASIS_SET"] = ("AUX_FIT " + auxiliar_basis)

    return sett


cp2k_pbe_guess = Settings(yaml.load("""
cp2k:
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
"""))

cp2k_pbe_main = Settings(yaml.load("""
cp2k:
  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional pbe: {}
      scf:
        eps_scf: 5e-4
        max_scf: 200
"""))


cp2k_pbe0_guess = Settings(yaml.load("""
cp2k:
   force_eval:
     subsys:
       cell:
         periodic: "None"
     dft:
       auxiliary_density_matrix_method:
         method: "basis_projection"
         admm_purification_method: "none"
       poisson:
         periodic: "None"
         psolver: "MT"
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

"""))


cp2k_pbe0_main = Settings(yaml.load("""
cp2k:
   force_eval:
     subsys:
       cell:
         periodic: "None"
     dft:
       auxiliary_density_matrix_method:
         method: "basis_projection"
         admm_purification_method: "none"
       poisson:
         periodic: "None"
         psolver: "MT"
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
           memory:
             max_memory: "5000"
             eps_storage_scaling: "0.1"
       scf:
          eps_scf: 5e-4
          max_scf: 200
"""))


kinds_template = Settings(yaml.load("""
cp2k:
  force_eval:
    subsys:
      kind:
        C:
          basis_set: DZVP-MOLOPT-SR-GTH-q4
          potential: GTH-PBE-q4
"""))


def generate_kinds(elements: list, basis: str, potential: str) -> Settings:
    """
    Generate the kind section for cp2k basis
    """
    s = Settings()
    subsys = s.cp2k.force_eval.subsys
    for e in elements:
        q = valence_electrons['-'.join(e, basis)]
        subsys.kind[e]['basis_set'] = "{}-q{}".format(basis, q)
        subsys.kind[e]['potential'] = "{}-q{}".format(potential, q)


# available templates
templates_dict = {
        "pbe_guess": cp2k_pbe_guess, "pbe_main": cp2k_pbe_main,
        "pbe0_guess": cp2k_pbe0_guess, "pbe0_main": cp2k_pbe0_main}


def create_settings_from_template(
        general: dict, template_name: str, path_traj_xyz: str) -> Settings:
    """
    Create a job Settings using the name provided by the user
    """
    setts = templates_dict[template_name]
    elements = read_unique_atomic_labels(path_traj_xyz)

    kinds = generate_kinds(elements, general['basis'], general['potential'])

    if 'pbe0' in template_name:
        return generate_auxiliar_basis(setts + kinds, general['basis'])
    else:
        return setts + kinds


def read_unique_atomic_labels(path_traj_xyz: str) -> set:
    """
    Return the unique atomic labels
    """
    mol = Molecule(path_traj_xyz, 'xyz')

    return set(at.symbol for at in mol.atoms)
