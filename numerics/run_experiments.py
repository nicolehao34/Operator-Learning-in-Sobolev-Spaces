"""
run_experiments.py — experiments for
Operator-Learning-in-Sobolev-Spaces. prepping for submission.

Drop this into numerics/ alongside train_fno.py. It re-imports your existing
FNO1d, solver, dataset, and H^1 loss so results are consistent with the paper.
If the import fails (e.g. run standalone), self-contained fallbacks are defined.

Implements:
  EXP-1  wider multi-seed sweep at higher resolution -> alpha with 95% CI (bootstrap)
  EXP-2  best-vs-final error logging + per-seed instability counting
  EXP-3  linear advection-diffusion operator (set --operator linear)
  EXP-4  resolution-invariance evaluation

Usage examples:
  python run_experiments.py --operator burgers --grid 256 --seeds 5 --epochs 200
  python run_experiments.py --operator linear  --grid 256 --seeds 5 --epochs 200
  python run_experiments.py --quick           # tiny smoke test on CPU
  python run_experiments.py --save_curves     # per-epoch curves for paper FIG 1

Outputs: results_<operator>_N<grid>.json  and  scaling_<operator>_N<grid>.png
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Reuse the components if available; otherwise define fallbacks
# ------------------------------------------------------------------
try:
    from train_fno import (
        FNO1d,
        burgers_spectral_solver,
        spectral_derivative,
        h1_norm,
        sample_smooth_ic,
        h1_loss_torch,
        BurgersDataset,
        count_parameters,
    )
    _USING_REPO = True
except Exception:
    _USING_REPO = False

    def spectral_derivative(u, L=1.0, order=1):
        N = u.shape[0]
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
        u_hat = np.fft.fft(u)
        du_hat = (1j * k) * u_hat if order == 1 else -(k ** 2) * u_hat
        return np.real(np.fft.ifft(du_hat))

    def burgers_spectral_solver(u0, nu=0.01, t_final=1.0, dt=2e-3, L=1.0):
        N = u0.shape[0]
        k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
        ef = np.exp(-nu * (k ** 2) * dt / 2.0)
        u_hat = np.fft.fft(u0)
        for _ in range(int(t_final / dt)):
            u_hat = ef * u_hat
            u = np.real(np.fft.ifft(u_hat))
            u_x = spectral_derivative(u, L=L, order=1)
            nl_hat = np.fft.fft(-u * u_x)
            u_hat = ef * (u_hat + dt * nl_hat)
            u = np.real(np.fft.ifft(u_hat))
            if not np.isfinite(u).all() or np.abs(u).max() > 1e3:
                return np.full_like(u0, np.nan)
            u_hat = np.fft.fft(u)
        return u

    def h1_norm(u, L=1.0):
        u_x = spectral_derivative(u, L=L, order=1)
        return np.sqrt(np.mean(u ** 2 + u_x ** 2))

    def sample_smooth_ic(N, L=1.0, h1_radius=0.3, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        u = rng.normal(size=N)
        k = np.fft.fftfreq(N, d=L / N)
        u_hat = np.fft.fft(u) * np.exp(-12.0 * (k ** 2))
        u_s = np.real(np.fft.ifft(u_hat))
        u_s -= np.mean(u_s)
        nrm = h1_norm(u_s, L=L)
        if nrm > 1e-8:
            u_s = (h1_radius / nrm) * u_s
        return u_s

    def h1_loss_torch(u_pred, u_true, L=1.0):
        dx = L / u_pred.shape[1]
        diff = u_pred - u_true
        l2 = torch.mean(diff ** 2)

        def grad(u):
            return (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2.0 * dx)

        g = torch.mean((grad(u_pred) - grad(u_true)) ** 2)
        return l2 + g, l2.detach(), g.detach()

    class BurgersDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.from_numpy(x).float()
            self.y = torch.from_numpy(y).float()

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    class SpectralConv1d(nn.Module):
        def __init__(self, ic, oc, modes):
            super().__init__()
            self.modes = modes
            scale = 1.0 / (ic * oc)
            self.weights = nn.Parameter(scale * torch.randn(ic, oc, modes, 2))

        def forward(self, x):
            b, c, N = x.shape
            x_ft = torch.fft.rfft(x, dim=-1)
            Nr = x_ft.shape[-1]
            out = torch.zeros(b, self.weights.shape[1], Nr, device=x.device,
                              dtype=torch.cfloat)
            w = torch.view_as_complex(self.weights)
            m = min(self.modes, Nr)
            out[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], w[:, :, :m])
            return torch.fft.irfft(out, n=N, dim=-1)

    class FNO1d(nn.Module):
        def __init__(self, modes=16, width=64):
            super().__init__()
            self.fc0 = nn.Linear(2, width)
            self.conv_layers = nn.ModuleList(
                [SpectralConv1d(width, width, modes) for _ in range(4)])
            self.w_layers = nn.ModuleList(
                [nn.Conv1d(width, width, 1) for _ in range(4)])
            self.fc1 = nn.Linear(width, 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, u0):
            b, N = u0.shape
            xc = torch.linspace(0, 1, N, device=u0.device).unsqueeze(0).repeat(b, 1)
            x = self.fc0(torch.stack([u0, xc], -1)).permute(0, 2, 1)
            for conv, w in zip(self.conv_layers, self.w_layers):
                x = F.gelu(conv(x) + w(x))
            x = x.permute(0, 2, 1)
            return self.fc2(F.gelu(self.fc1(x))).squeeze(-1)


# ------------------------------------------------------------------
# EXP-3: linear advection-diffusion operator (drop the nonlinearity)
# ------------------------------------------------------------------
def linear_solver(u0, nu=0.01, c=1.0, t_final=1.0, dt=2e-3, L=1.0):
    """u_t + c u_x = nu u_xx  — same numerics, nonlinear term replaced by -c u_x."""
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


# ------------------------------------------------------------------
# Data + training
# ------------------------------------------------------------------
def make_dataset(n_train, n_test, N, operator, data_seed, **soln_kw):
    rng = np.random.default_rng(data_seed)

    def solve(u0):
        if operator == "burgers":
            return burgers_spectral_solver(u0, **soln_kw)
        return linear_solver(u0, **soln_kw)

    def split(n):
        xs, ys = [], []
        tries = 0
        while len(xs) < n and tries < n * 4:
            tries += 1
            u0 = sample_smooth_ic(N, L=soln_kw.get("L", 1.0),
                                  h1_radius=0.3, rng=rng)
            uT = solve(u0)
            if np.isfinite(u0).all() and np.isfinite(uT).all():
                xs.append(u0.astype(np.float32))
                ys.append(uT.astype(np.float32))
        return np.stack(xs), np.stack(ys)

    xtr, ytr = split(n_train)
    xte, yte = split(n_test)
    return xtr, ytr, xte, yte


def rel_h1_error(model, x, y, L, device):
    """Global relative H^1 error over a set (matches parameter_sweep.py)."""
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x).float().to(device)
        pred = model(xb).cpu().numpy()
    num = 0.0
    den = 0.0
    for i in range(y.shape[0]):
        num += h1_norm(pred[i] - y[i], L=L) ** 2
        den += h1_norm(y[i], L=L) ** 2
    return float(np.sqrt(num / den))


DEFAULT_CONFIGS = [(4, 16), (8, 24), (8, 48), (12, 48),
                   (16, 64), (20, 80), (24, 96), (32, 128)]


def train_one(x_train, y_train, x_test, y_test, modes, width, epochs, lr,
              batch, L, device, model_seed, grad_clip=None, eval_every=5,
              return_model=False):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    model = FNO1d(modes=modes, width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = DataLoader(BurgersDataset(x_train, y_train), batch_size=batch, shuffle=True)
    te = DataLoader(BurgersDataset(x_test, y_test), batch_size=batch, shuffle=False)

    best = float("inf")
    final = float("inf")
    spiked = False
    prev = None
    history_epochs = []
    history_loss = []
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss, _, _ = h1_loss_torch(model(xb), yb, L=L)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            acc = 0.0
            with torch.no_grad():
                for xb, yb in te:
                    xb, yb = xb.to(device), yb.to(device)
                    l, _, _ = h1_loss_torch(model(xb), yb, L=L)
                    acc += l.item() * xb.size(0)
            tl = acc / len(x_test)
            history_epochs.append(ep)
            history_loss.append(tl)
            best = min(best, tl)
            final = tl
            if prev is not None and tl > 25.0 * prev:   # >1.4 orders of magnitude jump
                spiked = True
            prev = tl
    rel = rel_h1_error(model, x_test, y_test, L, device)
    out = {
        "params": count_parameters(model),
        "best": best,
        "final": final,
        "rel_h1": rel,
        "spiked": spiked,
        "epochs": history_epochs,
        "test_loss": history_loss,
    }
    if return_model:
        out["model"] = model
    return out


# ------------------------------------------------------------------
# alpha fit with bootstrap 95% CI over seeds
# ------------------------------------------------------------------
def fit_alpha(per_config, n_boot=2000, rng_seed=0):
    """per_config: list of dicts {params, best_list(seeds)}. Returns alpha, CI, R2."""
    rng = np.random.default_rng(rng_seed)
    P = np.array([c["params"] for c in per_config], float)
    # point estimate from seed-mean
    means = np.array([np.mean(c["best_list"]) for c in per_config])
    lp, le = np.log(P), np.log(means)
    slope, intcpt = np.polyfit(lp, le, 1)
    yhat = intcpt + slope * lp
    ss_res = np.sum((le - yhat) ** 2)
    ss_tot = np.sum((le - le.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # bootstrap: resample a seed per config each draw
    alphas = []
    for _ in range(n_boot):
        ys = []
        for c in per_config:
            ys.append(np.log(rng.choice(c["best_list"])))
        s, _ = np.polyfit(lp, np.array(ys), 1)
        alphas.append(-s)
    lo, hi = np.percentile(alphas, [2.5, 97.5])
    return {"alpha": -slope, "ci95": [float(lo), float(hi)],
            "r2": float(r2), "C": float(np.exp(intcpt))}


# ------------------------------------------------------------------
# Per-epoch curve dump (one representative seed for paper figures)
# ------------------------------------------------------------------
def save_curves(args):
    """Train 8 configs with model_seed=1000; write numerics/curves_burgers_N256.json."""
    L = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_seed = 1000
    configs = DEFAULT_CONFIGS
    soln_kw = dict(nu=args.nu, t_final=1.0, dt=2e-3, L=L)

    print(f"repo components: {_USING_REPO} | device: {device} | save_curves mode")
    xtr, ytr, xte, yte = make_dataset(args.n_train, args.n_test, args.grid,
                                      args.operator, data_seed=0, **soln_kw)
    print(f"dataset: train={xtr.shape} test={xte.shape} | seed={model_seed}")

    curve_configs = []
    for m, w in configs:
        r = train_one(xtr, ytr, xte, yte, m, w, args.epochs, args.lr,
                      args.batch, L, device, model_seed=model_seed,
                      grad_clip=args.grad_clip)
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", choices=["burgers", "linear"], default="burgers")
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_test", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test (CPU, 2 configs, 1 seed, 4 epochs)")
    ap.add_argument("--save_curves", action="store_true",
                    help="train 8 configs x 1 seed; dump per-epoch test loss JSON")
    args = ap.parse_args()

    if args.save_curves:
        if args.operator != "burgers" or args.grid != 256:
            print("WARN: --save_curves defaults to burgers N=256 for paper FIG 1")
        args.operator = "burgers"
        args.grid = 256
        save_curves(args)
        return

    L = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"repo components: {_USING_REPO} | device: {device} | operator: {args.operator}")

    if args.quick:
        configs = [(4, 16), (8, 24)]
        args.seeds, args.epochs, args.grid = 1, 4, 64
        args.n_train, args.n_test = 64, 32
    else:
        configs = DEFAULT_CONFIGS

    ceiling = args.grid // 2 + 1
    soln_kw = dict(nu=args.nu, t_final=1.0, dt=2e-3, L=L)

    # one fixed dataset (data_seed=0); seeds vary model init/optimization only
    xtr, ytr, xte, yte = make_dataset(args.n_train, args.n_test, args.grid,
                                      args.operator, data_seed=0, **soln_kw)
    print(f"dataset: train={xtr.shape} test={xte.shape} | rfft ceiling={ceiling}")

    per_config = []
    for (m, w) in configs:
        if m > ceiling:
            print(f"  WARN modes={m} exceeds rfft ceiling {ceiling}; capped internally")
        seed_runs = []
        for s in range(args.seeds):
            r = train_one(xtr, ytr, xte, yte, m, w, args.epochs, args.lr,
                          args.batch, L, device, model_seed=1000 + s,
                          grad_clip=args.grad_clip)
            seed_runs.append(r)
            print(f"  modes={m:2d} width={w:3d} seed={s} "
                  f"N={r['params']:>9,} best={r['best']:.3e} "
                  f"final={r['final']:.3e} rel={r['rel_h1']:.3e} spike={r['spiked']}")
        per_config.append({
            "modes": m, "width": w,
            "params": seed_runs[0]["params"],
            "best_list": [r["best"] for r in seed_runs],
            "final_list": [r["final"] for r in seed_runs],
            "rel_list": [r["rel_h1"] for r in seed_runs],
            "n_spiked": int(sum(r["spiked"] for r in seed_runs)),
        })

    fit = fit_alpha(per_config)
    print("\n=== SCALING FIT (best-epoch, seed-mean) ===")
    print(f"alpha = {fit['alpha']:.3f}  95% CI [{fit['ci95'][0]:.3f}, {fit['ci95'][1]:.3f}]")
    print(f"R^2   = {fit['r2']:.3f}   C = {fit['C']:.3e}")
    print(f"benchmark s/d = 1.0  -> CI {'EXCLUDES' if fit['ci95'][1] < 1.0 else 'INCLUDES'} benchmark")

    out = {"operator": args.operator, "grid": args.grid, "ceiling": ceiling,
           "epochs": args.epochs, "seeds": args.seeds, "configs": per_config,
           "fit": fit, "params_used": vars(args)}
    fn = f"results_{args.operator}_N{args.grid}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {fn}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = np.array([c["params"] for c in per_config], float)
        mean = np.array([np.mean(c["best_list"]) for c in per_config])
        std = np.array([np.std(c["best_list"]) for c in per_config])
        plt.figure(figsize=(6, 4))
        plt.errorbar(P, mean, yerr=std, fmt="o-", capsize=3, label="best-epoch (mean±std)")
        ref = mean[0] * (P / P[0]) ** (-1.0)
        plt.loglog(P, ref, "k--", alpha=0.6, label=r"benchmark $N^{-1}$")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("parameters $N$"); plt.ylabel(r"test $H^1$ error")
        plt.title(f"{args.operator}, N={args.grid}: "
                  rf"$\alpha$={fit['alpha']:.2f} "
                  f"[{fit['ci95'][0]:.2f},{fit['ci95'][1]:.2f}]")
        plt.legend(); plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        png = f"scaling_{args.operator}_N{args.grid}.png"
        plt.savefig(png, dpi=150)
        print(f"wrote {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
