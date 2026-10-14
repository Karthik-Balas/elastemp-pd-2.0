"""Command-line interface for the ElastEmp-PD research workflows."""
from __future__ import annotations

import argparse
import logging

from elastemp_pd import __version__

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elastemp-pd",
                                     description="Pressure-temperature phase diagram workflows")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--workspace", default=".", help="Calculation workspace (default: .)")
    commands = parser.add_subparsers(dest="command", required=True)

    enthalpy = commands.add_parser("prepare-enthalpy", help="Generate pressure-dependent VASP relaxations")
    enthalpy.add_argument("--structures", default="POSCARS")
    enthalpy.add_argument("--kpt-density", type=int, default=6000)
    enthalpy.add_argument("--p-min", type=float, required=True, help="GPa")
    enthalpy.add_argument("--p-max", type=float, required=True, help="GPa")
    enthalpy.add_argument("--num-p", type=int, required=True,
                          help="number of pressure intervals (gives num-p + 1 pressures)")
    enthalpy.add_argument("--encut", type=float, default=None,
                          help="plane-wave cutoff, eV (default: 1.3 x POTCAR ENMAX)")

    extract = commands.add_parser("extract-enthalpy", help="Read relaxations and compute H = E + PV")
    extract.add_argument("--allow-unconverged", action="store_true",
                         help="warn instead of failing when convergence, force or stress checks fail")
    extract.add_argument("--allow-enthalpy-mismatch", action="store_true",
                         help="warn instead of failing when E + PV disagrees with OUTCAR's enthalpy")
    extract.add_argument("--max-force", type=float, default=0.01, help="eV/A (default 0.01)")
    extract.add_argument("--max-stress", type=float, default=1.0,
                         help="max |residual pressure|, kBar (default 1.0)")

    phonon = commands.add_parser("prepare-phonon", help="Build DFPT supercells from relaxed structures")
    phonon.add_argument("--dim", nargs=3, type=int, default=[3, 3, 3], help="supercell (default 3 3 3)")
    phonon.add_argument("--no-primitive", action="store_true",
                        help="build supercells from the relaxed cell as-is instead of its primitive cell")
    phonon.add_argument("--kpt-density", type=int, default=1000)

    commands.add_parser("extract-phonon", help="Read DFPT force constants from vasprun.xml")

    thermal = commands.add_parser("thermal", help="Phonon thermal properties and G(T, P) tables")
    thermal.add_argument("--t-min", type=float, default=0)
    thermal.add_argument("--t-max", type=float, default=1000)
    thermal.add_argument("--t-step", type=float, default=10)
    thermal.add_argument("--mesh", nargs=3, type=int, default=[48, 48, 48], help="q-point mesh")
    thermal.add_argument("--plots", action="store_true")

    meta = commands.add_parser("metastable", help="Metastable pressure-temperature phase maps")
    meta.add_argument("--g-min", type=float, default=0, help="eV/atom")
    meta.add_argument("--g-max", type=float, default=0.2, help="eV/atom")
    meta.add_argument("--num-g", type=int, default=10, help="number of dG intervals")
    meta.add_argument("--fit-degree", type=int, default=None,
                      help="pressure-fit degree (default: min(3, n_pressures - 2))")
    meta.add_argument("--p-step", type=float, default=1.0, help="interpolated pressure step, GPa")
    meta.add_argument("--no-surrogate", action="store_true", help="skip the logistic-regression figure")
    meta.add_argument("--surrogate-c", type=float, default=None,
                      help="inverse L2 strength for the surrogate (default 100)")

    for sub in commands.choices.values():   # accept --workspace before or after the subcommand
        sub.add_argument("--workspace", default=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    ws = args.workspace
    try:
        if args.command == "prepare-enthalpy":
            from elastemp_pd.enthalpy.enthalpy_calc import prepare_enthalpy
            prepare_enthalpy(structures=args.structures, kptdensity=args.kpt_density, p_min=args.p_min,
                             p_max=args.p_max, num_p=args.num_p, workspace=ws, encut=args.encut)
        elif args.command == "extract-enthalpy":
            from elastemp_pd.enthalpy.enthalpy_calc import extract_enthalpy
            extract_enthalpy(workspace=ws, allow_unconverged=args.allow_unconverged,
                             allow_enthalpy_mismatch=args.allow_enthalpy_mismatch,
                             max_force=args.max_force, max_stress=args.max_stress)
        elif args.command == "prepare-phonon":
            from elastemp_pd.dynamic.dynamic_calc import prepare_phonon
            prepare_phonon(kptdensity=args.kpt_density, dim=args.dim, workspace=ws,
                           primitive=not args.no_primitive)
        elif args.command == "extract-phonon":
            from elastemp_pd.dynamic.dynamic_calc import extract_phonon
            extract_phonon(workspace=ws)
        elif args.command == "thermal":
            from elastemp_pd.thermal.thermal_calc import compute_thermal
            compute_thermal(t_min=args.t_min, t_max=args.t_max, t_step=args.t_step, mesh=args.mesh,
                            make_plots=args.plots, workspace=ws)
        elif args.command == "metastable":
            from elastemp_pd.ml.classifier import DEFAULT_C
            from elastemp_pd.thermal.metastable import compute_metastable
            compute_metastable(g_min=args.g_min, g_max=args.g_max, num_g=args.num_g, workspace=ws,
                               fit_degree=args.fit_degree, p_step=args.p_step,
                               surrogate=not args.no_surrogate,
                               C=DEFAULT_C if args.surrogate_c is None else args.surrogate_c)
    except Exception:
        logger.exception("Workflow failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
