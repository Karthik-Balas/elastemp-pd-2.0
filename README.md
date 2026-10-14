# ElastEmp-PD

Pressure–temperature phase diagrams and metastability maps from VASP relaxations and DFPT phonons.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
elastemp-pd --help
```

Requires Python ≥ 3.10 and phonopy ≥ 3.5 (installed from pip; tested on 3.5.1 and 4.8.3). VASP is external and not installed by pip.

## Workflow

Run from a dedicated workspace containing `POTCAR` and `POSCARS/POSCAR-<id>` (or pass `--workspace DIR`, before or after the subcommand):

```bash
elastemp-pd prepare-enthalpy --p-min 0 --p-max 100 --num-p 4   # 5 pressures: 0, 25, 50, 75, 100 GPa
#   ... run the VASP relaxations in Calcs/Structure-*/P-*/ ...
elastemp-pd extract-enthalpy                                   # checks each relaxation, then H = E + PV
elastemp-pd prepare-phonon                                     # 3x3x3 supercells of each relaxed primitive cell
#   ... run VASP DFPT in Calcs/Structure-*/P-*/Phonon_calculation/ ...
elastemp-pd extract-phonon                                     # FORCE_CONSTANTS from vasprun.xml
elastemp-pd thermal --t-min 0 --t-max 1000 --t-step 10 --plots
elastemp-pd metastable --g-min 0 --g-max 0.2 --num-g 10
```

## What is computed

For structure *i* at temperature *T* and pressure *P*, per atom:

```
G_i(T, P) = H_i(P) + F_vib,i(T; V_i(P))
H_i(P)    = E_i + P·V_i            (relaxed at P with PSTRESS; 1 GPa·Å³ = 1/160.218 eV)
F_vib     = phonopy free energy    (kJ/mol per cell → eV/atom with 1/96.485)
```

* The PV term is essential at GPa pressures: for graphite vs diamond at 10 GPa, P·ΔV ≈ 0.3 eV/atom using their zero-pressure volumes.
* Phonons are computed on the **relaxed cell at each pressure**, so F_vib carries the pressure dependence.
* That cell is reduced to its **primitive cell** before building the supercell. For diamond, 3×3×3 is then 54 atoms instead of 216; per-atom results don't depend on this choice. `--no-primitive` keeps the cell as relaxed. Species order is preserved, so the POTCAR still matches for compounds.
* Thermal expansion at fixed P is neglected: there is one volume per pressure, not a full quasi-harmonic treatment (QHA).
* Imaginary modes (below −0.1 THz) are reported as warnings. phonopy leaves them out of F_vib, so a structure that warns is dynamically unstable at that pressure.

### Input settings

* **ENCUT** defaults to 1.3 × the largest ENMAX in the POTCAR, rounded up to 10 eV (520 eV for carbon), or set it with `--encut`. The POTCAR default cutoff would leave cell relaxations with Pulay stress, an artificial stress from the finite basis that biases every volume and PV term. DFPT uses the same cutoff as the relaxation.
* **Supercell:** `--dim` defaults to 3 3 3.
* **q-mesh:** `--mesh` defaults to 48 48 48. F_vib is recomputed on a mesh half as dense, and a warning is logged if that changes it by more than 0.1 meV/atom.

### Relaxation checks (`extract-enthalpy`)

Every relaxation must pass all of these before its enthalpy is used. Any failure stops the run.

| Check | Default limit | Override |
|---|---|---|
| OUTCAR says "reached required accuracy" | n/a | `--allow-unconverged` |
| Final maximum force | 0.01 eV/Å | `--max-force`, `--allow-unconverged` |
| Final residual pressure ("external pressure", relative to PSTRESS) | 1 kB | `--max-stress`, `--allow-unconverged` |
| E + P·V(CONTCAR) vs OUTCAR's "enthalpy is TOTEN" | 1 meV/atom | `--allow-enthalpy-mismatch` |

All four values are recorded per relaxation in `static_enthalpy.csv`.

### Phase labels

With dE_i = G_i − min_j G_j and an energy window ΔG (eV/atom), every grid point gets:

* `equilibrium_phase`: the lowest-G structure (the same for every ΔG).
* `nearest_phase_above_window`: the structure with the smallest dE_i ≥ ΔG, which is the original notebook's metastability definition. It equals the equilibrium phase at ΔG = 0, and is `none` if no structure lies that high.
* `accessible_phases` / `n_accessible`: every structure with dE_i ≤ ΔG.

### Pressure interpolation and its validation

G(P) is interpolated at each T by a **least-squares** polynomial (default degree min(3, n_P − 2)). This is never an exact fit through the points. Its reliability is reported three ways:

1. **Fit residuals** at the computed pressures: `pressure_fit_residuals.csv`.
2. **Leave-one-pressure-out** errors at interior pressures: `pressure_fit_loo.csv`. This needs n_P ≥ degree + 3, so 6 pressures for a cubic; with 5, dropping one leaves a cubic that passes exactly through the rest.
3. **Method sensitivity:** every equilibrium transition pressure is found with both the polynomial and a shape-preserving PCHIP interpolant (`interpolation_sensitivity.csv`). A warning is logged if any transition moves by more than half the smallest pressure spacing, or appears with only one method.

Labels on the computed pressure grid, with no interpolation, are always written as well (`phase_labels_dG=*.csv`). They are the reference when the grid is sparse or the sensitivity check warns. With fewer than 3 pressures, only these are produced.

### Logistic-regression surrogate

`Metastable_map_dG=*.png` is the exact label map and is the result. `Metastable_slice_LogisticRegression_dG=*.png` is a quadratic, L2-regularized multinomial logistic fit to those labels. It can only smooth the exact boundaries, never improve them, so every fit reports the fraction of grid points it misclassifies and marks them on the figure. `metastable_summary.csv` holds this together with per-phase recall. Use `--no-surrogate` to skip it or `--surrogate-c` to change the regularization (default C = 100).

## Outputs

| Path | Content |
|---|---|
| `output/postprocessing/static_enthalpy.csv` | E, V, PV, H per atom, plus OUTCAR enthalpy difference, final force and residual pressure |
| `Calcs/.../Phonon_calculation/gibbs_temp.csv` | G, H, F_vib, S, C_v vs T; minimum phonon frequency; q-mesh check |
| `output/postprocessing/energy_temp_pressure_allphases.csv` | G(T, P) for all structures on the computed grid |
| `output/postprocessing/phase_labels_dG=<x>meV[_interpolated].csv` | equilibrium phase, nearest phase above the window, accessible sets |
| `output/postprocessing/pressure_fit_residuals.csv`, `pressure_fit_loo.csv`, `interpolation_sensitivity.csv` | interpolation checks |
| `output/postprocessing/metastable_summary.csv` | phases per ΔG, surrogate error |
| `output/results/*.png` | stability curves, equilibrium map, metastable maps, surrogate maps |



## License

MIT. Copyright © 2023, UChicago Argonne, LLC. See `LICENSE`.
