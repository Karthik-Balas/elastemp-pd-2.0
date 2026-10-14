"""Phonon thermal properties and G(T, P) tables.

G(T, P) per atom = H_static(P) + F_vib(T; V(P))

H_static comes from the relaxation at P and F_vib from harmonic phonons of that
relaxed cell. Thermal expansion at fixed P is neglected (no volume grid / QHA).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from elastemp_pd.core.units import KJMOL_TO_EV  # noqa: E402
from elastemp_pd.core.workspace import (  # noqa: E402
    ENTHALPY_FILE,
    POST,
    PREP,
    RESULTS,
    THERMAL_FILE,
    energy_column,
    pressure_dir,
    pressure_label,
    read_manifest,
    read_pressures,
    read_structures,
)
from elastemp_pd.dynamic.dynamic_calc import FC_FILE, PHONON_DIR, UNITCELL, build_phonopy  # noqa: E402

logger = logging.getLogger(__name__)

#: frequencies below this (THz) are reported as imaginary (acoustic noise near Gamma is ~1e-3)
IMAGINARY_TOL_THZ = -0.1
#: warn when halving the q-mesh changes F_vib by more than this (meV/atom) at any temperature
QMESH_TOL_MEV = 0.1


def _read_dim(folder: Path):
    dim_file = folder / "phonon_dim.txt"
    if not dim_file.is_file():
        raise FileNotFoundError(f"{dim_file} (re-run prepare-phonon)")
    return [int(x) for x in dim_file.read_text().split()]


def _thermal_properties(phonon) -> dict:
    """temperatures (K), free_energy (kJ/mol), entropy, heat_capacity (J/K/mol); per unit cell."""
    keys = ("temperatures", "free_energy", "entropy", "heat_capacity")
    result = phonon.thermal_properties
    if all(hasattr(result, k) for k in keys):          # phonopy >= 4
        return {k: np.asarray(getattr(result, k)) for k in keys}
    return {k: np.asarray(v) for k, v in phonon.get_thermal_properties_dict().items()}  # phonopy 3


def thermal_for_folder(folder: Path, *, t_min, t_max, t_step, mesh) -> pd.DataFrame:
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    folder = Path(folder)
    for name in (UNITCELL, FC_FILE, ENTHALPY_FILE):
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    phonon = build_phonopy(folder / UNITCELL, _read_dim(folder))
    phonon.force_constants = parse_FORCE_CONSTANTS(str(folder / FC_FILE))
    phonon.run_mesh(list(mesh), is_gamma_center=True)
    min_freq = float(np.min(phonon.mesh.frequencies))
    if min_freq < IMAGINARY_TOL_THZ:
        logger.warning("Imaginary modes (min %.3f THz) in %s: structure is dynamically unstable "
                       "here and F_vib ignores those modes", min_freq, folder)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    phonon.write_yaml_thermal_properties(filename=str(folder / "thermal_properties.yaml"))
    props = _thermal_properties(phonon)

    natoms = len(phonon.unitcell)
    h_static = float((folder / ENTHALPY_FILE).read_text().split()[0])
    f_vib = np.asarray(props["free_energy"]) * KJMOL_TO_EV / natoms     # kJ/mol per cell -> eV/atom

    # q-mesh convergence: repeat on a mesh half as dense and compare F_vib
    coarse = [max(1, int(m) // 2) for m in mesh]
    phonon.run_mesh(coarse, is_gamma_center=True)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    f_coarse = _thermal_properties(phonon)["free_energy"] * KJMOL_TO_EV / natoms
    qmesh_change = 1000 * float(np.max(np.abs(f_coarse - f_vib)))
    if qmesh_change > QMESH_TOL_MEV:
        logger.warning("q-mesh may be unconverged in %s: halving %s changes F_vib by up to %.3f meV/atom",
                       folder, "x".join(map(str, mesh)), qmesh_change)
    table = pd.DataFrame({
        "temperature": props["temperatures"],
        "gibbs_energy": h_static + f_vib,
        "enthalpy_static": h_static,
        "f_vib": f_vib,
        "entropy_J_per_K_mol_atom": np.asarray(props["entropy"]) / natoms,
        "cv_J_per_K_mol_atom": np.asarray(props["heat_capacity"]) / natoms,
        "min_frequency_THz": min_freq,
        "qmesh_halving_change_meV": qmesh_change,
    })
    table.to_csv(folder / THERMAL_FILE, index=False)
    return table


def compute_thermal(t_min=0, t_max=1000, t_step=10, mesh=(48, 48, 48), make_plots=False,
                    workspace=".") -> pd.DataFrame:
    workspace = Path(workspace).resolve()
    if t_step <= 0 or t_max < t_min or t_min < 0:
        raise ValueError("Require 0 <= t_min <= t_max and t_step > 0")
    folders = read_manifest(workspace / PREP / "filelist_dynamic.csv", workspace)
    (workspace / PREP / "input_parameters_temp.txt").write_text(
        f"T_min = {t_min}\nT_max = {t_max}\nT_step = {t_step}\nmesh = {' '.join(map(str, mesh))}\n")
    for folder in folders:
        logger.info("Computing thermal properties in %s", folder)
        thermal_for_folder(folder, t_min=t_min, t_max=t_max, t_step=t_step, mesh=mesh)
    return collect_gibbs(workspace, make_plots=make_plots)


def collect_gibbs(workspace, *, make_plots=False) -> pd.DataFrame:
    """Assemble ``energy_temp_pressure_allphases.csv``: temp, pressure, energy-Structure-<id> ..."""
    workspace = Path(workspace).resolve()
    structures = read_structures(workspace)
    pressures = read_pressures(workspace)
    frames = []
    for sid in structures:
        per_structure = []
        for p in pressures:
            path = pressure_dir(workspace, sid, p) / PHONON_DIR / THERMAL_FILE
            if not path.is_file():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            per_structure.append(pd.DataFrame({"temp": df["temperature"], "pressure": float(p),
                                               energy_column(sid): df["gibbs_energy"]}))
        table = pd.concat(per_structure, ignore_index=True)
        table.to_csv(workspace / "Calcs" / f"Structure-{sid}" / "energy_temp_pressure.csv", index=False)
        frames.append(table)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["temp", "pressure"], how="inner", validate="one_to_one")
    merged = merged[["temp", "pressure"] + [energy_column(s) for s in structures]]
    (workspace / POST).mkdir(parents=True, exist_ok=True)
    (workspace / RESULTS).mkdir(parents=True, exist_ok=True)
    merged.to_csv(workspace / POST / "energy_temp_pressure_allphases.csv", index=False)

    cols = [energy_column(s) for s in structures]
    for p in pressures:
        sub = merged[merged["pressure"] == float(p)].copy()
        energies = sub[cols]
        sub["e_min"] = energies.min(axis=1)
        sub["equilibrium_phase"] = [structures[i] for i in energies.to_numpy().argmin(axis=1)]
        sub.to_csv(workspace / POST / f"energy_temp_P={pressure_label(p)}_allphases.csv", index=False)
        if make_plots:
            fig, ax = plt.subplots()
            for sid, col in zip(structures, cols):
                ax.plot(sub["temp"], (sub[col] - sub["e_min"]) * 1000, label=sid, linewidth=2)
            ax.set(xlabel="Temperature (K)", ylabel="G - G_min (meV/atom)",
                   title=f"Relative stability, P = {pressure_label(p)} GPa")
            ax.legend(title="Structure")
            fig.savefig(workspace / RESULTS / f"Relative-stability-curves-P={pressure_label(p)}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    return merged
