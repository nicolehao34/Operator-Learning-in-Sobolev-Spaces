"""
run_experiments.py — multi-seed sweep + bootstrap-CI alpha fit.

See numerics/THEORY_TESTS.md for theory-test commands via run_theory_tests.py.
"""

import argparse
import json

import torch

from theory_runner import (
    DEFAULT_CONFIGS_1D,
    OPERATOR_META,
    make_dataset,
    run_sweep,
    train_one,
)


def save_curves(args):
    """Train 8 configs with model_seed=1000; dump per-epoch test loss JSON."""
    L = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_seed = 1000
    configs = DEFAULT_CONFIGS_1D
    soln_kw = dict(nu=args.nu, t_final=1.0, dt=2e-3, L=L)

    print(f"save_curves mode | device: {device}")
    xtr, ytr, xte, yte = make_dataset(
        args.n_train, args.n_test, args.grid, args.operator,
        data_seed=0, ic_smoothness=1.0, **soln_kw,
    )
    print(f"dataset: train={xtr.shape} test={xte.shape} | seed={model_seed}")

    curve_configs = []
    for m, w in configs:
        r = train_one(
            xtr, ytr, xte, yte, m, w, args.epochs, args.lr, args.batch,
            L, device, model_seed=model_seed, dim=1,
            grad_clip=args.grad_clip, lr_schedule=args.lr_schedule,
        )
        curve_configs.append({
            "modes": m, "width": w, "params": r["params"],
            "epochs": r["epochs"], "test_loss": r["test_loss"],
        })
        print(f"  modes={m:2d} width={w:3d} N={r['params']:>9,} "
              f"points={len(r['epochs'])}")

    out = {
        "operator": args.operator,
        "grid": args.grid,
        "model_seed": model_seed,
        "epochs": args.epochs,
        "eval_every": 5,
        "configs": curve_configs,
    }
    fn = f"curves_{args.operator}_N{args.grid}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {fn}")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", choices=list(OPERATOR_META.keys()), default="burgers")
    ap.add_argument("--grid", type=int, default=256)
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
                    help="Sobolev exponent s for IC sampler (default: operator meta)")
    ap.add_argument("--benchmark_exponent", type=float, default=None,
                    help="theory benchmark s/d (default: s/d from meta)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test (2 configs, 1 seed, 4 epochs)")
    ap.add_argument("--save_curves", action="store_true",
                    help="train 8 configs x 1 seed; dump per-epoch test loss JSON")
    ap.add_argument("--no_plot", action="store_true")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.plot = not args.no_plot

    if args.ic_smoothness is None:
        args.ic_smoothness = OPERATOR_META[args.operator]["s"]

    if args.save_curves:
        if args.operator != "burgers" or args.grid != 256:
            print("WARN: --save_curves defaults to burgers N=256 for paper FIG 1")
        args.operator = "burgers"
        args.grid = 256
        save_curves(args)
        return

    run_sweep(args)


if __name__ == "__main__":
    main()
