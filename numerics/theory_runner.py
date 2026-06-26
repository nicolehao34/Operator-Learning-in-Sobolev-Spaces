"""
Shared experiment sweep for theory tests (1D/2D operators, bootstrap alpha fit).

Imported by run_experiments.py and run_theory_tests.py.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from plot_style import versioned_output_path
from train_fno import (
    FNO1d,
    FNO2d,
    BurgersDataset,
    adv_diff_2d_solver,
    burgers_spectral_solver,
    count_parameters,
    estimate_ic_spectrum_decay,
    h1_loss_torch,
    h1_loss_torch_2d,
    h1_norm,
    h1_norm_2d,
    sample_smooth_ic,
    sample_smooth_ic_2d,
)

# ------------------------------------------------------------------
# Operator registry: dimension d, input smoothness s, benchmark s/d
# ------------------------------------------------------------------
OPERATOR_META = {
    "burgers": {"d": 1, "s": 1, "benchmark_exponent": 1.0, "dim": 1},
    "linear": {"d": 1, "s": 1, "benchmark_exponent": 1.0, "dim": 1},
    "adv_diff_2d": {"d": 2, "s": 1, "benchmark_exponent": 0.5, "dim": 2},
}

DEFAULT_CONFIGS_1D = [(4, 16), (8, 24), (8, 48), (12, 48),
                      (16, 64), (20, 80), (24, 96), (32, 128)]
DEFAULT_CONFIGS_2D = [(4, 16), (8, 24), (8, 32), (12, 32),
                      (12, 48), (16, 48), (16, 64), (20, 64)]
QUICK_CONFIGS_1D = [(4, 16), (8, 24)]
QUICK_CONFIGS_2D = [(4, 16), (8, 24)]


def linear_solver(u0, nu=0.01, c=1.0, t_final=1.0, dt=2e-3, L=1.0):
    """1D u_t + c u_x = nu u_xx (spectral integrating factor)."""
    from train_fno import spectral_derivative
    N = u0.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    ef = np.exp(-nu * (k ** 2) * dt / 2.0)
    u_hat = np.fft.fft(u0)
    for _ in range(int(t_final / dt)):
        u_hat = ef * u_hat
        u = np.real(np.fft.ifft(u_hat))
        u_x = spectral_derivative(u, L=L, order=1)
        u_hat = ef * (u_hat + dt * np.fft.fft(-c * u_x))
        u = np.real(np.fft.ifft(u_hat))
        u_hat = np.fft.fft(u)
    return u


def make_dataset(n_train, n_test, grid, operator, data_seed, ic_smoothness=1.0,
                 h1_radius=0.3, **soln_kw):
    meta = OPERATOR_META[operator]
    dim = meta["dim"]
    rng = np.random.default_rng(data_seed)
    L = soln_kw.get("L", 1.0)

    def solve(u0):
        if operator == "burgers":
            return burgers_spectral_solver(u0, **soln_kw)
        if operator == "adv_diff_2d":
            return adv_diff_2d_solver(u0, **soln_kw)
        return linear_solver(u0, **soln_kw)

    def sample_ic():
        if dim == 2:
            return sample_smooth_ic_2d(grid, L=L, h1_radius=h1_radius,
                                       ic_smoothness=ic_smoothness, rng=rng)
        return sample_smooth_ic(grid, L=L, h1_radius=h1_radius,
                              ic_smoothness=ic_smoothness, rng=rng)

    def split(n):
        xs, ys = [], []
        tries = 0
        while len(xs) < n and tries < n * 4:
            tries += 1
            u0 = sample_ic()
            uT = solve(u0)
            if np.isfinite(u0).all() and np.isfinite(uT).all():
                xs.append(u0.astype(np.float32))
                ys.append(uT.astype(np.float32))
        if len(xs) < n:
            raise RuntimeError(f"only generated {len(xs)}/{n} stable samples")
        return np.stack(xs), np.stack(ys)

    xtr, ytr = split(n_train)
    xte, yte = split(n_test)
    return xtr, ytr, xte, yte


def rel_h1_error(model, x, y, L, device, dim=1):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x).float().to(device)
        pred = model(xb).cpu().numpy()
    num, den = 0.0, 0.0
    norm_fn = h1_norm_2d if dim == 2 else h1_norm
    for i in range(y.shape[0]):
        num += norm_fn(pred[i] - y[i], L=L) ** 2
        den += norm_fn(y[i], L=L) ** 2
    return float(np.sqrt(num / den))


def train_one(x_train, y_train, x_test, y_test, modes, width, epochs, lr,
              batch, L, device, model_seed, dim=1, grad_clip=None,
              lr_schedule="none", eval_every=5, return_model=False):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    Model = FNO2d if dim == 2 else FNO1d
    loss_fn = h1_loss_torch_2d if dim == 2 else h1_loss_torch
    model = Model(modes=modes, width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr = DataLoader(BurgersDataset(x_train, y_train), batch_size=batch, shuffle=True)
    te = DataLoader(BurgersDataset(x_test, y_test), batch_size=batch, shuffle=False)

    best, final = float("inf"), float("inf")
    spiked, prev = False, None
    history_epochs, history_loss = [], []

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss, _, _ = loss_fn(model(xb), yb, L=L)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if scheduler is not None:
            scheduler.step()

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            acc = 0.0
            with torch.no_grad():
                for xb, yb in te:
                    xb, yb = xb.to(device), yb.to(device)
                    l, _, _ = loss_fn(model(xb), yb, L=L)
                    acc += l.item() * xb.size(0)
            tl = acc / len(x_test)
            history_epochs.append(ep)
            history_loss.append(tl)
            best = min(best, tl)
            final = tl
            if prev is not None and tl > 25.0 * prev:
                spiked = True
            prev = tl

    rel = rel_h1_error(model, x_test, y_test, L, device, dim=dim)
    out = {
        "params": count_parameters(model),
        "best": best,
        "final": final,
        "rel_h1": rel,
        "spiked": spiked,
        "epochs": history_epochs,
        "test_loss": history_loss,
        "lr_schedule": lr_schedule,
        "grad_clip": grad_clip,
    }
    if return_model:
        out["model"] = model
    return out


def fit_alpha(per_config, n_boot=2000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    P = np.array([c["params"] for c in per_config], float)
    means = np.array([np.mean(c["best_list"]) for c in per_config])
    lp, le = np.log(P), np.log(means)
    slope, intcpt = np.polyfit(lp, le, 1)
    yhat = intcpt + slope * lp
    ss_res = np.sum((le - yhat) ** 2)
    ss_tot = np.sum((le - le.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alphas = []
    for _ in range(n_boot):
        ys = [np.log(rng.choice(c["best_list"])) for c in per_config]
        s, _ = np.polyfit(lp, np.array(ys), 1)
        alphas.append(-s)
    lo, hi = np.percentile(alphas, [2.5, 97.5])
    return {"alpha": -slope, "ci95": [float(lo), float(hi)],
            "r2": float(r2), "C": float(np.exp(intcpt))}


def run_sweep(args):
    """Full multi-seed sweep; writes results JSON and optional scaling PNG."""
    meta = OPERATOR_META[args.operator]
    dim = meta["dim"]
    d = meta["d"]
    s = float(args.ic_smoothness)
    benchmark = (float(args.benchmark_exponent)
                 if args.benchmark_exponent is not None else s / d)

    L = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | operator: {args.operator} | d={d} s={s} "
          f"benchmark s/d={benchmark:.3g}")

    if args.quick:
        configs = QUICK_CONFIGS_2D if dim == 2 else QUICK_CONFIGS_1D
        args.seeds, args.epochs, args.grid = 1, 4, 64
        args.n_train, args.n_test = 32, 16
    else:
        configs = DEFAULT_CONFIGS_2D if dim == 2 else DEFAULT_CONFIGS_1D

    ceiling = args.grid // 2 + 1
    soln_kw = dict(nu=args.nu, t_final=1.0, dt=2e-3, L=L)
    if args.operator == "adv_diff_2d":
        soln_kw.update(cx=getattr(args, "cx", 1.0), cy=getattr(args, "cy", 1.0))
    elif args.operator == "linear":
        soln_kw["c"] = getattr(args, "c", 1.0)

    xtr, ytr, xte, yte = make_dataset(
        args.n_train, args.n_test, args.grid, args.operator,
        data_seed=0, ic_smoothness=s, h1_radius=0.3, **soln_kw,
    )
    print(f"dataset: train={xtr.shape} test={xte.shape} | rfft ceiling={ceiling}")

    if args.quick and dim == 1:
        u_sample = xtr[0]
        eff = estimate_ic_spectrum_decay(u_sample, L=L)
        print(f"IC spectrum diagnostic: ic_smoothness={s} -> effective decay ~{eff:.2f}")

    per_config = []
    total_spiked = 0
    for m, w in configs:
        if m > ceiling:
            print(f"  WARN modes={m} exceeds rfft ceiling {ceiling}")
        seed_runs = []
        for si in range(args.seeds):
            r = train_one(
                xtr, ytr, xte, yte, m, w, args.epochs, args.lr, args.batch,
                L, device, model_seed=1000 + si, dim=dim,
                grad_clip=args.grad_clip, lr_schedule=args.lr_schedule,
            )
            seed_runs.append(r)
            total_spiked += int(r["spiked"])
            print(f"  modes={m:2d} width={w:3d} seed={si} "
                  f"N={r['params']:>9,} best={r['best']:.3e} "
                  f"final={r['final']:.3e} spike={r['spiked']} "
                  f"clip={args.grad_clip} sched={args.lr_schedule}")
        per_config.append({
            "modes": m, "width": w,
            "params": seed_runs[0]["params"],
            "best_list": [r["best"] for r in seed_runs],
            "final_list": [r["final"] for r in seed_runs],
            "rel_list": [r["rel_h1"] for r in seed_runs],
            "n_spiked": int(sum(r["spiked"] for r in seed_runs)),
        })

    fit = fit_alpha(per_config)
    ci_excludes = fit["ci95"][1] < benchmark
    print("\n=== SCALING FIT (best-epoch, seed-mean) ===")
    print(f"alpha = {fit['alpha']:.3f}  95% CI [{fit['ci95'][0]:.3f}, {fit['ci95'][1]:.3f}]")
    print(f"R^2   = {fit['r2']:.3f}   C = {fit['C']:.3e}")
    print(f"benchmark s/d = {benchmark:.3g}  -> CI "
          f"{'EXCLUDES' if ci_excludes else 'INCLUDES'} benchmark")
    print(f"total n_spiked (all configs/seeds): {total_spiked}")

    out = {
        "operator": args.operator,
        "grid": args.grid,
        "d": d,
        "s": s,
        "benchmark_exponent": benchmark,
        "ceiling": ceiling,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "configs": per_config,
        "fit": fit,
        "total_n_spiked": total_spiked,
        "params_used": {
            k: v for k, v in vars(args).items()
            if not k.startswith("_")
        },
    }
    fn = f"results_{args.operator}_N{args.grid}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {fn}")

    if getattr(args, "plot", True):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            P = np.array([c["params"] for c in per_config], float)
            mean = np.array([np.mean(c["best_list"]) for c in per_config])
            std = np.array([np.std(c["best_list"]) for c in per_config])
            plt.figure(figsize=(6, 4))
            plt.errorbar(P, mean, yerr=std, fmt="o-", capsize=3,
                         label="best-epoch (mean±std)")
            ref = mean[0] * (P / P[0]) ** (-benchmark)
            plt.loglog(P, ref, "k--", alpha=0.6,
                       label=rf"benchmark $N^{{-{benchmark:.2g}}}$")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("parameters $N$")
            plt.ylabel(r"test $H^1$ error")
            plt.title(f"{args.operator}, N={args.grid}: "
                      rf"$\alpha$={fit['alpha']:.2f} "
                      f"[{fit['ci95'][0]:.2f},{fit['ci95'][1]:.2f}]")
            plt.legend()
            plt.grid(True, which="both", ls=":")
            plt.tight_layout()
            canonical = f"scaling_{args.operator}_N{args.grid}.png"
            png = versioned_output_path(os.path.join(os.getcwd(), canonical))
            plt.savefig(png, dpi=150)
            plt.close()
            print(f"wrote {png} (original {canonical} preserved)")
        except Exception as e:
            print(f"plot skipped: {e}")

    return out
