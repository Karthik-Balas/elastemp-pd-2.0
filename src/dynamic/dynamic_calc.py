"""Prepare and process pressure-dependent DFPT phonon calculations.

The phonon unit cell is the *relaxed* structure at each pressure (CONTCAR of the
relaxation), so force constants -- and therefore F_vib -- carry the pressure
dependence. 

By default the relaxed cell is first reduced to its primitive cell, so a 3x3x3
supercell of diamond is 54 atoms rather than 216 for the conventional cell.
Per-atom thermodynamics do not depend on that choice.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from elastemp_pd.core.makefiles import make_incar_dfpt, make_kpoints, read_incar_value
from elastemp_pd.core.workspace import ENTHALPY_FILE, PREP, read_manifest, write_manifest

logger = logging.getLogger(__name__)

PHONON_DIR = "Phonon_calculation"
UNITCELL = "POSCAR-unitcell"
FC_FILE = "FORCE_CONSTANTS"
DEFAULT_DIM = (3, 3, 3)


def _supercell_matrix(dim) -> np.ndarray:
    dim = [int(d) for d in dim]
    if len(dim) != 3 or min(dim) < 1:
        raise ValueError("dim must be three positive integers")
    return np.diag(dim)


def build_phonopy(unitcell_path: Path, dim):
    """Phonopy object whose 'primitive' cell is the unit cell itself.

    primitive_matrix='P' (identity) makes thermal properties per POSCAR-unitcell,
    matching the per-atom normalisation used downstream. Do not use None: from
    phonopy v4 it means 'auto', which silently switches to the reduced primitive
    cell (e.g. 2 instead of 8 atoms for diamond) and breaks the normalisation.
    """
    from phonopy import Phonopy
    from phonopy.interface.calculator import read_crystal_structure

    unitcell, _ = read_crystal_structure(str(unitcell_path), interface_mode="vasp")
    phonon = Phonopy(unitcell, supercell_matrix=_supercell_matrix(dim), primitive_matrix="P")
    if len(phonon.primitive) != len(phonon.unitcell):
        raise RuntimeError("phonopy primitive cell differs from the unit cell; normalisation would be wrong")
    return phonon


def write_supercell(unitcell_path: Path, dim, destination: Path) -> None:
    from phonopy.interface.vasp import write_vasp

    write_vasp(str(destination), build_phonopy(unitcell_path, dim).supercell)


def species_order(structure: Structure) -> list[str]:
    """Element order of the POSCAR species line (must match the POTCAR)."""
    order = []
    for site in structure:
        if site.species_string not in order:
            order.append(site.species_string)
    return order


def phonon_unitcell(relaxed: Structure, *, primitive: bool = True) -> Structure:
    """Relaxed cell, optionally reduced to its primitive cell with species order preserved."""
    if not primitive:
        return relaxed
    prim = relaxed.get_primitive_structure()
    order = species_order(relaxed)
    prim = Structure.from_sites(sorted(prim.sites, key=lambda site: order.index(site.species_string)))
    if species_order(prim) != order:
        raise RuntimeError("Primitive reduction changed the species order; POTCAR would not match")
    v_rel, v_prim = relaxed.volume / relaxed.num_sites, prim.volume / prim.num_sites
    if abs(v_prim - v_rel) > 1e-6 * v_rel:
        raise RuntimeError("Primitive reduction changed the volume per atom")
    return prim


def prepare_phonon(kptdensity=1000, dim=DEFAULT_DIM, workspace=".", *, primitive: bool = True) -> list[Path]:
    workspace = Path(workspace).resolve()
    folders = read_manifest(workspace / PREP / "filelist_enthalpy.csv", workspace)
    targets = []
    for folder in folders:
        for name in (ENTHALPY_FILE, "CONTCAR", "POTCAR", "INCAR"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"{folder / name} (run extract-enthalpy first)")
        encut = read_incar_value(folder / "INCAR", "ENCUT")
        if encut is None:
            raise ValueError(f"No ENCUT in {folder / 'INCAR'}; the DFPT run must use the relaxation cutoff")
        target = folder / PHONON_DIR
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / ENTHALPY_FILE, target / ENTHALPY_FILE)
        shutil.copy2(folder / "POTCAR", target / "POTCAR")
        relaxed = Structure.from_file(folder / "CONTCAR")          # relaxed geometry at this pressure
        unit = phonon_unitcell(relaxed, primitive=primitive)
        unit.to(filename=str(target / UNITCELL), fmt="poscar")
        write_supercell(target / UNITCELL, dim, target / "POSCAR")
        make_kpoints(Structure.from_file(target / "POSCAR"), kptdensity, file_loc=target)
        make_incar_dfpt(target, encut=float(encut))
        (target / "phonon_dim.txt").write_text(" ".join(str(int(d)) for d in dim) + "\n")
        targets.append(target)
        logger.info("Prepared phonon calculation in %s (%d-atom cell, %s supercell)", target,
                    unit.num_sites, "x".join(map(str, dim)))
    write_manifest(targets, workspace / PREP / "filelist_dynamic.csv", workspace)
    return targets


def read_dfpt_force_constants(folder: Path) -> np.ndarray:
    """Force constants (eV/A^2, supercell x supercell) from a DFPT vasprun.xml."""
    from phonopy.interface.vasp import parse_force_constants

    vasprun = Path(folder) / "vasprun.xml"
    if not vasprun.is_file():
        raise FileNotFoundError(vasprun)
    force_constants, _ = parse_force_constants(vasprun)
    if force_constants is None:
        raise ValueError(f"No Hessian found in {vasprun} (was IBRION=8 used?)")
    return force_constants


def extract_phonon(workspace=".") -> list[Path]:
    from phonopy.interface.vasp import write_FORCE_CONSTANTS

    workspace = Path(workspace).resolve()
    folders = read_manifest(workspace / PREP / "filelist_dynamic.csv", workspace)
    for folder in folders:
        logger.info("Extracting force constants in %s", folder)
        write_FORCE_CONSTANTS(read_dfpt_force_constants(folder), filename=str(folder / FC_FILE))
    return folders
