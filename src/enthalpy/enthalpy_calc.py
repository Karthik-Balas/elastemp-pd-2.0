"""Generate pressure-dependent VASP relaxations and extract static enthalpies.

Phase stability at pressure P is decided by the enthalpy H = E + PV (and, with
phonons, G = H + F_vib). Comparing bare energies E across phases with different
volumes drops the first-order pressure term and misplaces every transition.

Every relaxation is checked before its enthalpy is used:

* VASP printed "reached required accuracy";
* the final maximum force is below ``max_force`` (eV/A);
* the final residual ("external") pressure is below ``max_stress`` (kBar). With
  PSTRESS set, VASP reports the external pressure relative to the target, so it
  should be ~0 at convergence;
* H = E + P*V(CONTCAR) agrees with VASP's own "enthalpy is TOTEN" line to
  ``ENTHALPY_TOL`` eV/atom.

Failures raise by default; ``allow_unconverged`` / ``allow_enthalpy_mismatch``
downgrade them to warnings.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from elastemp_pd.core.makefiles import default_encut, make_incar_relax, make_kpoints
from elastemp_pd.core.units import GPA_A3_TO_EV
from elastemp_pd.core.workspace import (
    ENTHALPY_FILE,
    POST,
    PREP,
    pressure_dir,
    pressure_from_folder,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)

RELAX_CONVERGED = "reached required accuracy"
ENTHALPY_TOL = 1e-3     # eV/atom
DEFAULT_MAX_FORCE = 0.01   # eV/A
DEFAULT_MAX_STRESS = 1.0   # kBar (0.1 GPa)

_NUM = r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_OUTCAR_ENTHALPY = re.compile(r"enthalpy is\s+TOTEN\s*=\s*" + _NUM)
_EXTERNAL_PRESSURE = re.compile(r"external pressure\s*=\s*" + _NUM + r"\s*kB")


def pressure_grid(p_min: float, p_max: float, num_p: int) -> list[float]:
    """``num_p`` intervals between ``p_min`` and ``p_max`` -> ``num_p + 1`` pressures (GPa)."""
    if num_p < 0:
        raise ValueError("num_p must be >= 0")
    if num_p == 0:
        return [round(float(p_min), 6)]
    if p_min >= p_max:
        raise ValueError("p_max must be greater than p_min")
    return [round(float(p), 6) for p in np.linspace(p_min, p_max, num_p + 1)]


def prepare_enthalpy(structures="POSCARS", kptdensity=6000, p_min=0.0, p_max=0.0, num_p=0,
                     workspace=".", *, encut: float | None = None) -> list[Path]:
    """Create one relaxation folder per (structure, pressure).

    ``encut`` defaults to 1.3 x the largest ENMAX in the POTCAR, rounded up to 10 eV.
    """
    workspace = Path(workspace).resolve()
    root = Path(structures)
    if not root.is_absolute():
        root = workspace / root
    potcar = workspace / "POTCAR"
    if not potcar.is_file():
        raise FileNotFoundError(potcar)
    indices = sorted(p.name.removeprefix("POSCAR-") for p in root.glob("POSCAR-*") if p.is_file())
    if not indices:
        raise ValueError(f"No POSCAR-* files in {root}")
    pressures = pressure_grid(p_min, p_max, num_p)
    if encut is None:
        encut = default_encut(potcar)
        logger.info("ENCUT = %g eV (1.3 x POTCAR ENMAX)", encut)

    prep = workspace / PREP
    prep.mkdir(parents=True, exist_ok=True)
    (prep / "input_parameters.txt").write_text(
        f"P_min : {p_min} GPa\nP_max : {p_max} GPa\nnumber of pressure intervals : {num_p}\n"
        f"k-point density : {kptdensity}\nENCUT : {encut:g} eV\n")
    pd.Series(pressures).to_csv(prep / "pressures.csv", header=False, index=False)
    pd.Series(indices, dtype=str).to_csv(prep / "structures.csv", header=False, index=False)

    folders = []
    for index in indices:
        poscar = root / f"POSCAR-{index}"
        structure = Structure.from_file(poscar)
        for pressure in pressures:
            folder = pressure_dir(workspace, index, pressure)
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(poscar, folder / "POSCAR")
            shutil.copy2(potcar, folder / "POTCAR")
            make_kpoints(structure, kptdensity, file_loc=folder)
            make_incar_relax(pressure, file_loc=folder, encut=encut)
            folders.append(folder)
            logger.info("Prepared %s", folder)
    write_manifest(folders, prep / "filelist_enthalpy.csv", workspace)
    return folders


# ----------------------------------------------------------------------------- readers

def read_final_energy(folder: Path) -> float:
    """Final-step free energy F (eV, whole cell) from OSZICAR. Excludes the PV term."""
    oszicar = Path(folder) / "OSZICAR"
    lines = [ln for ln in oszicar.read_text().splitlines() if "F=" in ln]
    if not lines:
        raise ValueError(f"No completed ionic step in {oszicar}")
    tokens = lines[-1].split()
    return float(tokens[tokens.index("F=") + 1])


def _outcar_text(folder: Path) -> str:
    outcar = Path(folder) / "OUTCAR"
    return outcar.read_text(errors="replace") if outcar.is_file() else ""


def relaxation_converged(folder: Path) -> bool:
    return RELAX_CONVERGED in _outcar_text(folder)


def read_outcar_enthalpy(folder: Path) -> float | None:
    """VASP's own final 'enthalpy is TOTEN' (eV, whole cell), if printed."""
    matches = _OUTCAR_ENTHALPY.findall(_outcar_text(folder))
    return float(matches[-1]) if matches else None


def read_final_external_pressure(folder: Path) -> float | None:
    """Last 'external pressure' (kBar) in OUTCAR: the residual relative to PSTRESS."""
    matches = _EXTERNAL_PRESSURE.findall(_outcar_text(folder))
    return float(matches[-1]) if matches else None


def read_final_max_force(folder: Path) -> float | None:
    """Largest atomic force norm (eV/A) in the last TOTAL-FORCE block of OUTCAR."""
    lines = _outcar_text(folder).splitlines()
    starts = [i for i, ln in enumerate(lines) if "TOTAL-FORCE" in ln]
    if not starts:
        return None
    forces = []
    for ln in lines[starts[-1] + 2:]:        # skip header and the dashed rule
        if ln.strip().startswith("---") or not ln.strip():
            break
        values = ln.split()
        forces.append([float(x) for x in values[3:6]])
    if not forces:
        return None
    return float(np.max(np.linalg.norm(np.asarray(forces), axis=1)))


# ----------------------------------------------------------------------------- checks

def enthalpy_for_folder(folder: Path, *, allow_unconverged: bool = False,
                        allow_enthalpy_mismatch: bool = False,
                        max_force: float = DEFAULT_MAX_FORCE,
                        max_stress: float = DEFAULT_MAX_STRESS) -> dict:
    """H = E + P V for the relaxed structure in ``folder`` (per atom, eV), after checks."""
    folder = Path(folder)
    pressure = pressure_from_folder(folder)
    contcar = folder / "CONTCAR"
    if not contcar.is_file():
        raise FileNotFoundError(contcar)

    force = read_final_max_force(folder)
    ext_p = read_final_external_pressure(folder)
    problems = []
    if not relaxation_converged(folder):
        problems.append("VASP did not report 'reached required accuracy' (check NSW/EDIFFG)")
    if force is None:
        problems.append("no TOTAL-FORCE block in OUTCAR")
    elif force > max_force:
        problems.append(f"max force {force:.4f} eV/A > {max_force} eV/A")
    if ext_p is None:
        problems.append("no 'external pressure' line in OUTCAR")
    elif abs(ext_p) > max_stress:
        problems.append(f"residual pressure {ext_p:+.2f} kB exceeds {max_stress} kB")
    if problems:
        msg = f"Relaxation not converged in {folder}: " + "; ".join(problems)
        if not allow_unconverged:
            raise RuntimeError(msg)
        logger.warning(msg)

    relaxed = Structure.from_file(contcar)
    natoms = relaxed.num_sites
    energy = read_final_energy(folder)
    pv = pressure * relaxed.volume * GPA_A3_TO_EV
    enthalpy = energy + pv

    vasp_h = read_outcar_enthalpy(folder)
    h_diff = None if vasp_h is None else (enthalpy - vasp_h) / natoms
    if vasp_h is None:
        if pressure != 0:
            logger.warning("No 'enthalpy is TOTEN' line in %s; H = E + PV not cross-checked", folder)
    elif abs(h_diff) > ENTHALPY_TOL:
        msg = (f"H = E + PV ({enthalpy:.6f} eV) differs from OUTCAR enthalpy ({vasp_h:.6f} eV) by "
               f"{1000 * h_diff:+.2f} meV/atom in {folder}; check that the folder pressure matches "
               "PSTRESS and that CONTCAR is the final structure")
        if not allow_enthalpy_mismatch:
            raise RuntimeError(msg)
        logger.warning(msg)

    return {
        "structure": folder.parent.name.removeprefix("Structure-"),
        "pressure": pressure,
        "natoms": natoms,
        "volume_per_atom": relaxed.volume / natoms,
        "energy_per_atom": energy / natoms,
        "pv_per_atom": pv / natoms,
        "enthalpy_per_atom": enthalpy / natoms,
        "outcar_enthalpy_diff_meV": np.nan if h_diff is None else 1000 * h_diff,
        "max_force_eV_A": np.nan if force is None else force,
        "residual_pressure_kB": np.nan if ext_p is None else ext_p,
    }


def extract_enthalpy(workspace=".", *, allow_unconverged: bool = False,
                     allow_enthalpy_mismatch: bool = False, max_force: float = DEFAULT_MAX_FORCE,
                     max_stress: float = DEFAULT_MAX_STRESS) -> pd.DataFrame:
    """Write ``enthalpy_per_atom.txt`` in every relaxation folder and a summary table."""
    workspace = Path(workspace).resolve()
    folders = read_manifest(workspace / PREP / "filelist_enthalpy.csv", workspace)
    rows = []
    for folder in folders:
        row = enthalpy_for_folder(folder, allow_unconverged=allow_unconverged,
                                  allow_enthalpy_mismatch=allow_enthalpy_mismatch,
                                  max_force=max_force, max_stress=max_stress)
        (folder / ENTHALPY_FILE).write_text(f"{row['enthalpy_per_atom']:.10f}\n")
        rows.append(row)
        logger.info("Structure %s, P=%g GPa: H = %.6f eV/atom (PV = %.6f)", row["structure"],
                    row["pressure"], row["enthalpy_per_atom"], row["pv_per_atom"])
    table = pd.DataFrame(rows)
    (workspace / POST).mkdir(parents=True, exist_ok=True)
    table.to_csv(workspace / POST / "static_enthalpy.csv", index=False)
    return table
