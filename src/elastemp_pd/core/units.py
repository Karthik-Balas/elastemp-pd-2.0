"""Unit conversions used across the workflow (CODATA 2018)."""

#: 1 GPa * 1 Angstrom^3 expressed in eV  (1 eV/A^3 = 160.21766208 GPa)
GPA_A3_TO_EV = 1.0 / 160.21766208

#: 1 kJ/mol expressed in eV per particle (phonopy reports free energies in kJ/mol)
KJMOL_TO_EV = 1.0 / 96.48533212

#: VASP PSTRESS is in kBar; 1 GPa = 10 kBar
GPA_TO_KBAR = 10.0
