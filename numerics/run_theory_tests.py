#!/usr/bin/env python3
"""
Unified driver for theory-test experiments (vary d, s, operator class).

Each experiment is a single documented command; see numerics/THEORY_TESTS.md.
"""

import argparse

from theory_runner import OPERATOR_META, run_sweep


def build_parser():
    ap = argparse.ArgumentParser(
        description="Launch theory-test sweeps with benchmark-aware metadata.",
    )
    ap.add_argument(
        "--operator",
        choices=list(OPERATOR_META.keys()),
        default="burgers",
        help="PDE operator: burgers | linear | adv_diff_2d",
    )
    ap.add_argument("--grid", type=int, default=None,
                    help="grid points (1D length or 2D NxN side)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_test", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--lr_schedule", choices=["none", "cosine"], default="none")
    ap.add_argument("--ic_smoothness", type=float, default=None,
                    help="Sobolev exponent s for IC sampler")
    ap.add_argument("--benchmark_exponent", type=float, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test only")
    ap.add_argument("--no_plot", action="store_true")
    return ap


def default_grid(operator, quick):
    if quick:
        return 64
    if operator == "adv_diff_2d":
        return 64
    return 256


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.plot = not args.no_plot

    meta = OPERATOR_META[args.operator]
    if args.grid is None:
        args.grid = default_grid(args.operator, args.quick)
    if args.ic_smoothness is None:
        args.ic_smoothness = meta["s"]
    if args.benchmark_exponent is None:
        args.benchmark_exponent = args.ic_smoothness / meta["d"]

    print(f"=== theory test: {args.operator} | d={meta['d']} s={args.ic_smoothness} "
          f"benchmark={args.benchmark_exponent:.3g} ===")
    run_sweep(args)
    print(f"CAP OK: {args.operator} sweep completed")


if __name__ == "__main__":
    main()
