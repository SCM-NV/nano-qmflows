
__all__ = ["cp2k_pbe", "cp2k_pbe0"]

from qmflows.settings import Settings
import yaml


def generate_auxiliar_basis(sett: Settings, auxiliar_basis: str) -> Settings:
    """
    Generate the `auxiliar_basis` for all the atoms in the `sett`
    """
    kind = sett.cp2k.force_eval.subsys.kind
    for atom in kinds.keys():
        kind[atom]["BASIS_SET"] = "AUX_FIT " + auxiliar_basis

    return sett


pbe = Settings(yaml.load("""
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

pbe0 = Settings(yaml.load("""
cp2k:
   force_eval:
     subsys:
       cell:
         periodic: "None"
       kind:
         Cd:
            basis_set: "DZVP-MOLOPT-SR-GTH"
            Basis_set: "AUX_FIT SZV-MOLOPT-SR-GTH"
            potential: "GTH-PBE-q12"
         Se:
            basis_set: "DZVP-MOLOPT-SR-GTH"
            Basis_set: "AUX_FIT SZV-MOLOPT-SR-GTH"
            potential: "GTH-PBE-q6"

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

cp2k_pbe = pbe + kinds
cp2k_pbe0 = generate_auxiliar_basis(pbe0 + kinds, "SZV-MOLOPT-SR-GTH")
