"""Metastable P-T phase maps.

For every (T, P) point the relative energy of structure i is
``dE_i = G_i - min_j G_j`` (eV/atom). Per point and energy window ``dG``:

* ``equilibrium_phase`` -- the lowest-G structure (independent of dG).
* ``nearest_phase_above_window`` -- the structure whose dE_i is the smallest
  value >= dG (the original notebook's metastability definition; equal to the
  equilibrium phase at dG = 0). ``"none"`` when no structure lies that high.
* ``accessible_phases`` / ``n_accessible`` -- every structure with dE_i <= dG.

Pressure interpolation
----------------------
G(P) at each T is fitted by a least-squares polynomial of degree <= n_P - 2
(never an exact interpolant). Its reliability is reported three ways:

* fit residuals at the computed pressures (``pressure_fit_residuals.csv``);
* leave-one-pressure-out errors, when n_P >= degree + 3 (``pressure_fit_loo.csv``);
* sensitivity of every equilibrium transition pressure to the interpolation
  method -- polynomial vs. shape-preserving PCHIP (``interpolation_sensitivity.csv``).

Labels on the computed pressure grid (no interpolation) are always written too
and remain the primary reference for sparse grids.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from elastemp_pd.core.workspace import (
    POST,
    RESULTS,
    energy_column,
    read_structures,
    structure_from_column,
)
from elastemp_pd.ml.classifier import (
    DEFAULT_C,
    fit_phase_classifier,
    per_class_recall,
    surrogate_disagreement,
)
from elastemp_pd.plotting import NO_PHASE, plot_phase_map

logger = logging.getLogger(__name__)

TOL = 1e-9  # eV/atom; guards exact ties such as the ground state at dG = 0
LABEL_COLUMNS = ["equilibrium_phase", "nearest_phase_above_window", "accessible_phases", "n_accessible"]


def dg_values(g_min: float, g_max: float, num_g: int) -> np.ndarray:
    if g_min < 0 or g_max < g_min:
        raise ValueError("Require 0 <= g_min <= g_max")
    if num_g < 0:
        raise ValueError("num_g must be >= 0")
    return np.linspace(g_min, g_max, num_g + 1) if num_g else np.array([float(g_min)])


def dg_tag(dg: float) -> str:
    """Unambiguous file tag in meV/atom, e.g. 0.02 -> '20meV', 0.0125 -> '12.5meV'."""
    return f"{round(dg * 1000, 6):g}meV"


def phase_labels(energies: pd.DataFrame, dg: float) -> pd.DataFrame:
    """Label each row of an energy table (columns ``energy-Structure-<id>``)."""
    cols = list(energies.columns)
    ids = np.array([structure_from_column(c) for c in cols])
    values = energies.to_numpy(dtype=float)
    rel = values - values.min(axis=1, keepdims=True)

    above = rel - dg
    eligible = above >= -TOL
    choice = np.where(eligible, above, np.inf).argmin(axis=1)
    within = rel <= dg + TOL

    out = pd.DataFrame(rel, columns=cols, index=energies.index)
    out["equilibrium_phase"] = ids[values.argmin(axis=1)]
    out["nearest_phase_above_window"] = np.where(eligible.any(axis=1), ids[choice], NO_PHASE)
    out["accessible_phases"] = [";".join(ids[row]) for row in within]
    out["n_accessible"] = within.sum(axis=1)
    return out


# ----------------------------------------------------------------------------- interpolation

def _check_grid(df, structures):
    pressures = np.sort(df["pressure"].unique())
    temps = np.sort(df["temp"].unique())
    tables = {}
    for sid in structures:
        table = df.pivot(index="temp", columns="pressure", values=energy_column(sid)).loc[temps, pressures]
        if table.isna().any().any():
            raise ValueError(f"Missing (T, P) points for structure {sid}")
        tables[sid] = table.to_numpy()
    return pressures, temps, tables


def _resolve_degree(n_p: int, degree: int | None) -> int:
    if degree is None:
        degree = min(3, n_p - 2)
    if degree < 1 or n_p < degree + 2:
        raise ValueError(f"A degree-{degree} least-squares fit needs at least {degree + 2} pressures "
                         f"(have {n_p}); add pressures or lower --fit-degree")
    return degree


def _fitter(method: str, degree: int):
    if method == "poly":
        return lambda x, y: np.polynomial.Polynomial.fit(x, y, degree)
    if method == "pchip":
        return PchipInterpolator
    raise ValueError(f"Unknown interpolation method {method!r}")


def interpolate_pressure(df: pd.DataFrame, structures, *, degree: int | None = None,
                         p_step: float = 1.0, method: str = "poly") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate G(P) at every temperature onto a ``p_step`` grid.

    ``method='poly'``: least-squares polynomial (default). ``method='pchip'``:
    shape-preserving piecewise cubic through the computed points. Interpolation
    only -- the grid spans [min P, max P]. Returns the table and fit residuals.
    """
    pressures, temps, tables = _check_grid(df, structures)
    degree = _resolve_degree(len(pressures), degree)
    if p_step <= 0:
        raise ValueError("p_step must be positive")
    grid = np.round(np.arange(pressures[0], pressures[-1] + p_step / 2, p_step), 6)
    fit = _fitter(method, degree)

    columns = {"temp": np.repeat(temps, len(grid)), "pressure": np.tile(grid, len(temps))}
    residual_rows = []
    for sid in structures:
        fitted = []
        for t, row in zip(temps, tables[sid]):
            curve = fit(pressures, row)
            residual_rows.append({"structure": sid, "temp": t,
                                  "max_abs_residual_meV": 1000 * np.max(np.abs(curve(pressures) - row))})
            fitted.append(curve(grid))
        columns[energy_column(sid)] = np.concatenate(fitted)
    return pd.DataFrame(columns), pd.DataFrame(residual_rows)


def leave_one_pressure_out(df: pd.DataFrame, structures, *, degree: int | None = None) -> pd.DataFrame:
    """Predict each interior pressure from a fit to the others (needs n_P >= degree + 3)."""
    pressures, temps, tables = _check_grid(df, structures)
    degree = _resolve_degree(len(pressures), degree)
    if len(pressures) < degree + 3:
        raise ValueError(f"Leave-one-out with degree {degree} needs >= {degree + 3} pressures "
                         f"(have {len(pressures)})")
    rows = []
    for sid in structures:
        for t, row in zip(temps, tables[sid]):
            for i in range(1, len(pressures) - 1):          # interior points only: no extrapolation
                keep = np.arange(len(pressures)) != i
                poly = np.polynomial.Polynomial.fit(pressures[keep], row[keep], degree)
                rows.append({"structure": sid, "temp": t, "pressure_left_out": pressures[i],
                             "error_meV": 1000 * (poly(pressures[i]) - row[i])})
    return pd.DataFrame(rows)


def transition_pressures(dense: pd.DataFrame, cols) -> pd.DataFrame:
    """Equilibrium transitions along P at each T: (temp, transition 'a->b', pressure midpoint)."""
    ids = np.array([structure_from_column(c) for c in cols])
    rows = []
    for t, sub in dense.sort_values(["temp", "pressure"]).groupby("temp"):
        labels = ids[sub[cols].to_numpy().argmin(axis=1)]
        p = sub["pressure"].to_numpy()
        for k in np.nonzero(labels[1:] != labels[:-1])[0]:
            rows.append({"temp": t, "transition": f"{labels[k]}->{labels[k + 1]}",
                         "pressure": 0.5 * (p[k] + p[k + 1])})
    return pd.DataFrame(rows, columns=["temp", "transition", "pressure"])


def interpolation_sensitivity(poly: pd.DataFrame, pchip: pd.DataFrame, cols) -> pd.DataFrame:
    """Match transitions found with each method (by T, type and order) and report the shift."""
    a, b = transition_pressures(poly, cols), transition_pressures(pchip, cols)
    for frame in (a, b):
        frame["k"] = frame.groupby(["temp", "transition"]).cumcount()
    merged = a.merge(b, on=["temp", "transition", "k"], how="outer", suffixes=("_poly", "_pchip"))
    merged["shift_GPa"] = (merged["pressure_poly"] - merged["pressure_pchip"]).abs()
    return merged.drop(columns="k").sort_values(["temp", "transition"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- driver

def _label_and_save(df: pd.DataFrame, cols, dg: float, suffix: str, post: Path) -> pd.DataFrame:
    labels = phase_labels(df[cols], dg)
    tag = dg_tag(dg)
    df[["temp", "pressure"]].join(labels).to_csv(post / f"relative_energies_dG={tag}{suffix}.csv", index=False)
    df[["temp", "pressure"]].join(labels[LABEL_COLUMNS]).to_csv(post / f"phase_labels_dG={tag}{suffix}.csv",
                                                                index=False)
    return labels


def _validate_interpolation(raw, dense, structures, cols, *, degree, p_step, post) -> None:
    pressures = np.sort(raw["pressure"].unique())
    try:
        loo = leave_one_pressure_out(raw, structures, degree=degree)
        loo.to_csv(post / "pressure_fit_loo.csv", index=False)
        logger.info("Leave-one-pressure-out: worst error %.3f meV/atom", loo["error_meV"].abs().max())
    except ValueError as exc:
        logger.info("Leave-one-pressure-out skipped: %s", exc)

    pchip, _ = interpolate_pressure(raw, structures, degree=degree, p_step=p_step, method="pchip")
    sens = interpolation_sensitivity(dense, pchip, cols)
    sens.to_csv(post / "interpolation_sensitivity.csv", index=False)
    if sens.empty:
        logger.info("Interpolation check: no equilibrium transitions inside the pressure range")
        return
    tolerance = 0.5 * float(np.min(np.diff(pressures)))
    unmatched = sens["shift_GPa"].isna().sum()
    worst = sens["shift_GPa"].max()
    if unmatched or worst > tolerance:
        logger.warning("Transition pressures depend on the interpolation method (worst shift %.2f GPa, "
                       "%d transitions found by only one method). Trust the computed-grid labels and "
                       "add pressures near the boundary.", 0 if np.isnan(worst) else worst, unmatched)
    else:
        logger.info("Interpolation check: polynomial and PCHIP transitions agree within %.2f GPa", worst)


def compute_metastable(g_min=0.0, g_max=0.2, num_g=10, workspace=".", *, fit_degree=None,
                       p_step=1.0, surrogate=True, C=DEFAULT_C) -> pd.DataFrame:
    workspace = Path(workspace).resolve()
    post, results = workspace / POST, workspace / RESULTS
    source = post / "energy_temp_pressure_allphases.csv"
    if not source.is_file():
        raise FileNotFoundError(f"{source} (run thermal first)")
    dgs = dg_values(g_min, g_max, num_g)
    structures = read_structures(workspace)
    cols = [energy_column(s) for s in structures]
    raw = pd.read_csv(source)
    missing = set(cols) - set(raw.columns)
    if missing:
        raise ValueError(f"{source} lacks columns {sorted(missing)}")

    try:
        dense, residuals = interpolate_pressure(raw, structures, degree=fit_degree, p_step=p_step)
        residuals.to_csv(post / "pressure_fit_residuals.csv", index=False)
        dense.to_csv(post / "energy_temp_pressure_allphases_interpolated.csv", index=False)
        logger.info("Pressure fit: worst residual %.3f meV/atom", residuals["max_abs_residual_meV"].max())
        _validate_interpolation(raw, dense, structures, cols, degree=fit_degree, p_step=p_step, post=post)
        suffix = "_interpolated"
    except ValueError as exc:
        logger.warning("Skipping pressure interpolation: %s. Maps use the computed grid.", exc)
        dense, suffix = raw, ""

    classes = list(structures) + [NO_PHASE]
    equilibrium = phase_labels(dense[cols], 0.0)["equilibrium_phase"].to_numpy()
    plot_phase_map(dense["temp"], dense["pressure"], equilibrium, classes,
                   results / "Equilibrium_phase_map.png", title="Equilibrium phase (lowest G)")

    X = dense[["temp", "pressure"]].to_numpy(dtype=float)
    metrics = []
    for dg in dgs:
        raw_labels = _label_and_save(raw, cols, dg, "", post)
        labels = _label_and_save(dense, cols, dg, suffix, post) if suffix else raw_labels
        y = labels["nearest_phase_above_window"].to_numpy()
        tag = dg_tag(dg)
        phases = sorted(set(y))
        plot_phase_map(dense["temp"], dense["pressure"], y, classes,
                       results / f"Metastable_map_dG={tag}.png",
                       title=f"Nearest phase at or above ΔG = {tag}/atom")
        row = {"dG_meV": round(dg * 1000, 6), "n_phases": len(phases), "phases": ";".join(phases),
               "max_accessible": int(labels["n_accessible"].max()),
               "surrogate_disagreement": np.nan, "surrogate_recall": ""}
        if surrogate and len(phases) >= 2:
            model = fit_phase_classifier(X, y, C=C)
            pred = model.predict(X)
            row["surrogate_disagreement"] = surrogate_disagreement(model, X, y)
            row["surrogate_recall"] = json.dumps(per_class_recall(model, X, y))
            plot_phase_map(dense["temp"], dense["pressure"], pred, classes,
                           results / f"Metastable_slice_LogisticRegression_dG={tag}.png",
                           title=(f"Quadratic logistic surrogate, ΔG = {tag}/atom "
                                  f"({100 * row['surrogate_disagreement']:.2f}% of grid misclassified)"),
                           mismatch=pred != y)
            logger.info("dG=%s: surrogate misclassifies %.2f%% of grid points", tag,
                        100 * row["surrogate_disagreement"])
        elif surrogate:
            logger.info("dG=%s: single label (%s) everywhere; no surrogate fitted", tag, phases[0])
        metrics.append(row)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(post / "metastable_summary.csv", index=False)
    return metrics
