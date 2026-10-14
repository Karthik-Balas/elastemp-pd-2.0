"""Workspace layout helpers shared by every workflow step.

Layout::

    <workspace>/
        POTCAR
        POSCARS/POSCAR-<id>
        Calcs/Structure-<id>/P-<p>/                 relaxation at pressure p (GPa)
        Calcs/Structure-<id>/P-<p>/Phonon_calculation/
        output/preprocessing/   manifests and run parameters
        output/postprocessing/  tables
        output/results/         figures
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PREP = Path("output/preprocessing")
POST = Path("output/postprocessing")
RESULTS = Path("output/results")

ENTHALPY_FILE = "enthalpy_per_atom.txt"   # static H = E + PV per atom (eV), written by extract-enthalpy
THERMAL_FILE = "gibbs_temp.csv"           # G(T) per atom at one pressure, written by thermal


def pressure_label(p: float) -> str:
    """Canonical text form of a pressure, used in folder names (``P-<label>``)."""
    p = round(float(p), 6)
    return f"{p:g}"


def pressure_from_folder(folder: Path) -> float:
    name = Path(folder).name
    if not name.startswith("P-"):
        raise ValueError(f"Not a pressure folder: {folder}")
    return float(name.removeprefix("P-"))


def structure_dir(workspace: Path, structure_id: str) -> Path:
    return Path(workspace) / "Calcs" / f"Structure-{structure_id}"


def pressure_dir(workspace: Path, structure_id: str, pressure: float) -> Path:
    return structure_dir(workspace, structure_id) / f"P-{pressure_label(pressure)}"


def write_manifest(paths, manifest: Path, workspace: Path) -> None:
    """Store paths relative to the workspace so the workspace can be moved."""
    workspace = Path(workspace).resolve()
    rel = [str(Path(p).resolve().relative_to(workspace)) for p in paths]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(rel, dtype=str).to_csv(manifest, header=False, index=False)


def read_manifest(manifest: Path, workspace: Path) -> list[Path]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    workspace = Path(workspace).resolve()
    entries = pd.read_csv(manifest, header=None, dtype=str).iloc[:, 0]
    return [Path(p) if Path(p).is_absolute() else workspace / p for p in entries]


def read_structures(workspace: Path) -> list[str]:
    path = Path(workspace) / PREP / "structures.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None, dtype=str).iloc[:, 0].tolist()


def read_pressures(workspace: Path) -> np.ndarray:
    path = Path(workspace) / PREP / "pressures.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)


def energy_column(structure_id: str) -> str:
    return f"energy-Structure-{structure_id}"


def structure_from_column(column: str) -> str:
    return column.removeprefix("energy-Structure-")
