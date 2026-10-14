"""Phase-map figures with one fixed colour per phase across every figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NO_PHASE = "none"
_NONE_COLOR = (0.85, 0.85, 0.85, 1.0)


def phase_palette(classes):
    """Discrete colormap + norm so class ``i`` is always ``colors[i]`` in every artist.

    Uses tab10 (the matplotlib line-colour cycle), so structure ``i`` has the same
    colour here as in the relative-stability curves.
    """
    base = plt.get_cmap("tab10").colors
    colors = [(_NONE_COLOR if str(c) == NO_PHASE else (*base[i % len(base)], 1.0))
              for i, c in enumerate(classes)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))
    return cmap, norm, colors


def _grid(temps, pressures, codes):
    frame = pd.DataFrame({"temp": temps, "pressure": pressures, "code": codes})
    grid = frame.pivot(index="pressure", columns="temp", values="code")
    return grid.columns.to_numpy(float), grid.index.to_numpy(float), grid.to_numpy(float)


def plot_phase_map(temps, pressures, labels, classes, output_path, *, title, mismatch=None):
    """Draw a categorical T-P map on its native grid.

    ``classes`` fixes the colour order (pass all structures so colours are stable
    across dG slices). ``mismatch`` marks points to highlight (e.g. surrogate errors).
    """
    classes = [str(c) for c in classes]
    index = {c: i for i, c in enumerate(classes)}
    labels = np.asarray(labels).astype(str)
    unknown = set(labels) - set(index)
    if unknown:
        raise ValueError(f"Labels not in class list: {sorted(unknown)}")
    codes = np.array([index[lab] for lab in labels])
    cmap, norm, colors = phase_palette(classes)
    t, p, z = _grid(temps, pressures, codes)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.pcolormesh(t, p, z, cmap=cmap, norm=norm, shading="nearest")
    handles = [Patch(color=colors[index[c]], label=c) for c in classes if c in set(labels)]
    if mismatch is not None and np.any(mismatch):
        mismatch = np.asarray(mismatch, bool)
        ax.scatter(np.asarray(temps)[mismatch], np.asarray(pressures)[mismatch], s=14, c="k",
                   marker="x", linewidths=0.8)
        handles.append(Line2D([], [], color="k", marker="x", linestyle="none", label="misclassified"))
    ax.set(xlabel="Temperature (K)", ylabel="Pressure (GPa)", title=title)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), title="Phase")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
