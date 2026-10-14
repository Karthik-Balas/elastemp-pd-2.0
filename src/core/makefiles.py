"""VASP input-file generation and small INCAR/POTCAR readers."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from elastemp_pd.core.units import GPA_TO_KBAR

#: ENCUT = ENCUT_FACTOR * max(ENMAX): the usual margin against Pulay stress in cell relaxations
ENCUT_FACTOR = 1.3

_ENMAX = re.compile(r"ENMAX\s*=\s*([\d.]+)")


def potcar_enmax(potcar) -> float | None:
    """Largest ENMAX (eV) over all species in a POTCAR, or None if absent."""
    values = [float(v) for v in _ENMAX.findall(Path(potcar).read_text(errors="replace"))]
    return max(values) if values else None


def default_encut(potcar) -> float:
    enmax = potcar_enmax(potcar)
    if enmax is None:
        raise ValueError(f"No ENMAX found in {potcar}; pass --encut explicitly")
    return float(np.ceil(ENCUT_FACTOR * enmax / 10) * 10)   # e.g. 400 eV -> 520 eV


def read_incar_value(incar, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{key}\s*=\s*([^\s!#]+)", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(Path(incar).read_text())
    return match.group(1) if match else None


def make_incar_relax(p_gpa: float, file_loc, *, encut: float) -> Path:
    """INCAR for a full (ions + cell) relaxation under external pressure ``p_gpa``.

    PSTRESS is in kBar (1 GPa = 10 kBar). ENCUT is set explicitly because the
    POTCAR default leaves cell relaxations with Pulay stress, biasing V and PV.
    """
    path = Path(file_loc) / "INCAR"
    path.write_text(
        f"PREC = Accurate\nENCUT = {encut:g}\nICHARG = 0\nNWRITE = 0\nLORBIT = 11\n"
        "EDIFF = 1e-6\nIBRION = 2\nISIF = 3\nNSW = 100\nEDIFFG = 1e-4\nPOTIM = 0.015\n"
        "ISMEAR = -5\nSIGMA = 0.01\nISPIN = 2\nISYM = 2\nSYMPREC = 1e-4\n"
        "LCHARG = .FALSE.\nLWAVE = .FALSE.\nNPAR = 4\n"
        f"PSTRESS = {p_gpa * GPA_TO_KBAR:g}\n"
    )
    return path


def make_incar_dfpt(file_loc, *, encut: float) -> Path:
    """INCAR for a DFPT (IBRION=8) force-constant calculation.

    DFPT does not move atoms or the cell, so PSTRESS has no effect and is omitted:
    the pressure enters through the relaxed geometry the supercell is built from.
    ENCUT matches the relaxation so both use the same basis.
    """
    path = Path(file_loc) / "INCAR"
    path.write_text(
        f"PREC = Accurate\nENCUT = {encut:g}\nALGO = Fast\nICHARG = 0\nNWRITE = 0\nLORBIT = 11\n"
        "LREAL = .FALSE.\nADDGRID = .TRUE.\nEDIFF = 1e-6\nIBRION = 8\n"
        "ISMEAR = -5\nSIGMA = 0.01\nISPIN = 2\nISYM = 2\nSYMPREC = 1e-4\n"
        "LCHARG = .FALSE.\nLWAVE = .FALSE.\n"
    )
    return path


def kpoint_mesh(struct, kptdensity: float) -> tuple[int, int, int]:
    """Gamma-centred mesh with ~``kptdensity`` k-points per reciprocal atom.

    Each subdivision is rounded, made even (by subtracting one), and clamped to >= 1
    -- the same rule as the original notebook.
    """
    nkpts = kptdensity / struct.num_sites
    rec = 2 * np.pi * np.transpose(struct.lattice.inv_matrix)
    b = np.linalg.norm(rec, axis=1)
    step = (np.prod(b) / nkpts) ** (1.0 / 3)
    mesh = []
    for bi in b:
        n = int(round(bi / step))
        if n % 2:
            n -= 1
        mesh.append(max(n, 1))
    return tuple(mesh)


def make_kpoints(struct, kptdensity: float, file_loc=".") -> Path:
    n1, n2, n3 = kpoint_mesh(struct, kptdensity)
    path = Path(file_loc) / "KPOINTS"
    path.write_text(f"kpt_density={kptdensity:g}\n0\nGamma\n{n1} {n2} {n3}\n0. 0. 0.\n")
    return path
