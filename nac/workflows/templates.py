
__all__ = ["cp2k_pbe_guess", "cp2k_pbe_main", "cp2k_pbe0_guess", "cp2k_pbe0_main"]

from qmflows.settings import Settings
import yaml


def generate_auxiliar_basis(sett: Settings, auxiliar_basis: str) -> Settings:
    """
    Generate the `auxiliar_basis` for all the atoms in the `sett`
    """
    kind = sett.cp2k.force_eval.subsys.kind
    for atom in kind.keys():
        kind[atom]["BASIS_SET"] = ("AUX_FIT " + auxiliar_basis)

    return sett


pbe_guess = Settings(yaml.load("""
cp2k:
  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional: "pbe"
      scf:
        eps_scf: 1e-6
        added_mos: 0
        scf_guess: "restart"
        ot:
          minimizer: "DIIS"
          n_diis: 7
          preconditioner: "FULL_SINGLE_INVERSE"
"""))

pbe_main = Settings(yaml.load("""
cp2k:
  force_eval:
    subsys:
      cell:
        periodic: "None"
    dft:
      xc:
        xc_functional: "pbe"
      scf:
        eps_scf: 5e-4
        max_scf: 200
"""))


pbe0_guess = Settings(yaml.load("""
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
          eps_scf: 1e-6
          added_mos: 0
          scf_guess: "restart"
          ot:
            minimizer: "DIIS"
            n_diis: 7
            preconditioner: "FULL_SINGLE_INVERSE"

"""))


pbe0_main = Settings(yaml.load("""
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

kinds = Settings(yaml.load("""
cp2k:
  force_eval:
    subsys:
      kind:
        Br:
          basis_set: DZVP-MOLOPT-SR-GTH-q7
          potential: GTH-PBE-q7
        Ca:
          basis_set: DZVP-MOLOPT-SR-GTH-q10
          potential: GTH-PBE-q10
        Cd:
          BASIS_SET: DZVP-MOLOPT-SR-GTH-q12
          POTENTIAL: GTH-PBE-q12
        Cl:
          basis_set: DZVP-MOLOPT-SR-GTH-q7
          potential: GTH-PBE-q7
        Cs:
          basis_set: DZVP-MOLOPT-SR-GTH-q9
          potential: GTH-PBE-q9
        I:
          basis_set: DZVP-MOLOPT-SR-GTH-q7
          potential: GTH-PBE-q7
        Pb:
          basis_set: DZVP-MOLOPT-SR-GTH-q4
          potential: GTH-PBE-q4
        Sn:
          basis_set:  DZVP-MOLOPT-SR-GTH-q4
          potential:  GTH-PBE-q4

"""))

cp2k_pbe_guess = pbe_guess + kinds
cp2k_pbe_main = pbe_main + kinds
cp2k_pbe0_guess = generate_auxiliar_basis(pbe0_guess + kinds, "SZV-MOLOPT-SR-GTH")
cp2k_pbe0_main = generate_auxiliar_basis(pbe0_main + kinds, "SZV-MOLOPT-SR-GTH")
