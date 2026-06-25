"""
Regenerate paper figures from saved experiment JSON (no full sweep re-run).

Reads numerics/results_*.json and numerics/curves_burgers_N256.json (FIG 1).
Writes PNGs to the repo root for LaTeX \\includegraphics.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_paper_style, format_param_count, viridis_by_size

NUMERICS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(NUMERICS_DIR)
MIN_PNG_BYTES = 5 * 1024


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save_fig(path):
    plt.savefig(path)
    plt.close()


def _assert_png(path):
    assert os.path.isfile(path), f"missing: {path}"
    size = os.path.getsize(path)
    assert size > MIN_PNG_BYTES, f"{path} too small ({size} bytes)"


def _sorted_configs(results):
    return sorted(results["configs"], key=lambda c: c["params"])


def fig_learning_curves(curves_path, out_path):
    curves = _load_json(curves_path)
    configs = sorted(curves["configs"], key=lambda c: c["params"])
    colors = viridis_by_size(len(configs))

    fig, ax = plt.subplots()
    legend_lines = []
    for i, cfg in enumerate(configs):
        ep = np.array(cfg["epochs"], dtype=float)
        loss = np.array(cfg["test_loss"], dtype=float)
        (line,) = ax.plot(ep, loss, color=colors[i], label=format_param_count(cfg["params"]))
        legend_lines.append(line)
        # Honesty check: plotted data must match JSON exactly.
        assert np.array_equal(ep, cfg["epochs"])
        assert np.allclose(loss, cfg["test_loss"])

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"test $H^1$ loss")
    ax.set_title(r"Test $H^1$ loss vs epoch (N=256, 8 model sizes)")
    ax.legend(handles=legend_lines, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(right=0.72)
    _save_fig(out_path)
    assert len(legend_lines) == 8
    return "8 lines, log y-axis"


def fig_instability(results_path, out_path):
    results = _load_json(results_path)
    configs = _sorted_configs(results)
    seeds = results["seeds"]
    colors = viridis_by_size(len(configs))

    params = np.array([c["params"] for c in configs], float)
    fracs = [c["n_spiked"] / seeds for c in configs]

    fig, ax = plt.subplots()
    for p, frac, col, c in zip(params, fracs, colors, configs):
        expected = c["n_spiked"] / seeds
        assert abs(frac - expected) < 1e-12
        ax.bar(p, frac, width=p * 0.35, color=col, edgecolor="0.3", linewidth=0.5,
               align="center")
        ax.text(p, frac + 0.03, f"{c['n_spiked']}/{seeds}", ha="center", va="bottom",
                fontsize=9)

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("parameters $N$")
    ax.set_ylabel("fraction of seeds unstable")
    ax.set_title("Optimization instability vs model size (N=256)")
    _save_fig(out_path)
    return f"8 bars, heights={fracs}"


def fig_best_vs_final(results_path, out_path):
    results = _load_json(results_path)
    configs = _sorted_configs(results)

    P = np.array([c["params"] for c in configs], float)
    best_mean = np.array([np.mean(c["best_list"]) for c in configs])
    final_mean = np.array([np.mean(c["final_list"]) for c in configs])
    # Per-seed min/max error bars (std can go negative on a log axis and clip).
    best_lo = np.array([np.min(c["best_list"]) for c in configs])
    best_hi = np.array([np.max(c["best_list"]) for c in configs])
    final_lo = np.array([np.min(c["final_list"]) for c in configs])
    final_hi = np.array([np.max(c["final_list"]) for c in configs])
    best_yerr = np.array([best_mean - best_lo, best_hi - best_mean])
    final_yerr = np.array([final_mean - final_lo, final_hi - final_mean])

    assert np.all(final_mean >= best_mean - 1e-15 * np.maximum(best_mean, 1.0))

    all_vals = []
    for c in configs:
        all_vals.extend(c["best_list"])
        all_vals.extend(c["final_list"])
    ylo, yhi = float(min(all_vals)), float(max(all_vals))

    fig, ax = plt.subplots()
    ax.errorbar(P, best_mean, yerr=best_yerr, fmt="o-", capsize=3, label="best-epoch")
    ax.errorbar(P, final_mean, yerr=final_yerr, fmt="s-", capsize=3, label="final-epoch")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Pad one log-decade beyond observed range so error caps are not clipped.
    ax.set_ylim(ylo / 10.0, yhi * 10.0)
    ax.set_xlabel("parameters $N$")
    ax.set_ylabel(r"test $H^1$ error")
    ax.set_title("Best- vs final-epoch error: optimization-limited regime")
    ax.legend()
    _save_fig(out_path)
    return "2 series, log-log"


def fig_alpha_resolution(path64, path256, out_path):
    r64 = _load_json(path64)
    r256 = _load_json(path256)

    grids = [r64["grid"], r256["grid"]]
    alphas = [r64["fit"]["alpha"], r256["fit"]["alpha"]]
    ci_lo = [r64["fit"]["ci95"][0], r256["fit"]["ci95"][0]]
    ci_hi = [r64["fit"]["ci95"][1], r256["fit"]["ci95"][1]]
    yerr = [
        [alphas[0] - ci_lo[0], ci_hi[0] - alphas[0]],
        [alphas[1] - ci_lo[1], ci_hi[1] - alphas[1]],
    ]

    fig, ax = plt.subplots()
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.6, label=r"benchmark $s/d=1$")
    ax.errorbar(grids, alphas, yerr=np.array(yerr).T, fmt="o", capsize=5, label=r"fitted $\alpha$")
    ax.set_xscale("log")
    ax.set_xticks(grids)
    ax.set_xticklabels([str(g) for g in grids])
    ax.set_xlabel("grid resolution")
    ax.set_ylabel(r"fitted exponent $\alpha$")
    ax.set_title("Scaling exponent is resolution-robust")
    ax.legend()
    _save_fig(out_path)
    return "2 points + benchmark line, log x-axis"


def fig_scaling_burgers(results_path, out_path):
    """Scaling plot matching original layout; fonts from apply_paper_style()."""
    results = _load_json(results_path)
    configs = _sorted_configs(results)
    fit = results["fit"]

    P = np.array([c["params"] for c in configs], float)
    mean = np.array([np.mean(c["best_list"]) for c in configs])
    std = np.array([np.std(c["best_list"]) for c in configs])

    fig, ax = plt.subplots()
    ax.errorbar(P, mean, yerr=std, fmt="o-", capsize=3, label="best-epoch (mean±std)")
    ref = mean[0] * (P / P[0]) ** (-1.0)
    ax.loglog(P, ref, "k--", alpha=0.6, label=r"benchmark $N^{-1}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters $N$")
    ax.set_ylabel(r"test $H^1$ error")
    ax.set_title(
        f"{results['operator']}, N={results['grid']}: "
        rf"$\alpha$={fit['alpha']:.2f} "
        f"[{fit['ci95'][0]:.2f},{fit['ci95'][1]:.2f}]"
    )
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    _save_fig(out_path)
    return "1 series + benchmark, log-log"


def fig_longrun(longrun_path, out_path):
    data = _load_json(longrun_path)
    runs = sorted(data["runs"], key=lambda r: r["max_epochs"])
    colors = viridis_by_size(len(runs))

    fig, ax = plt.subplots()
    for run, col in zip(runs, colors):
        ep = np.array(run["epochs"], dtype=float)
        loss = np.array(run["test_loss"], dtype=float)
        ax.plot(ep, loss, color=col, label=f"{run['max_epochs']} epochs")

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"test $H^1$ loss")
    ax.set_title(r"Long-run learning curves (modes=24, width=96, $N=64$)")
    ax.legend()
    _save_fig(out_path)
    return f"{len(runs)} lines, log y-axis"


def fig_prediction(pred_path, out_path):
    d = _load_json(pred_path)
    x = np.array(d["x"])
    u0 = np.array(d["u0"])
    uT_true = np.array(d["uT_true"])
    uT_pred = np.array(d["uT_pred"])
    g_true = np.array(d["grad_true"])
    g_pred = np.array(d["grad_pred"])

    c_true = viridis_by_size(3)[1]
    c_pred = viridis_by_size(3)[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 4.0))
    ax1.plot(x, u0, color="0.5", alpha=0.7, label=r"$u(x,0)$")
    ax1.plot(x, uT_true, color=c_true, label=r"$u(x,1)$ true")
    ax1.plot(x, uT_pred, "--", color=c_pred, label=r"$u(x,1)$ pred")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u$")
    ax1.set_title("Solution at $t=1$")
    ax1.legend(fontsize=8)

    ax2.plot(x, g_true, color=c_true, label=r"$\partial_x u$ true")
    ax2.plot(x, g_pred, "--", color=c_pred, label=r"$\partial_x u$ pred")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"$\partial_x u$")
    ax2.set_title("Derivative at $t=1$")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(out_path)
    return "2 panels, linear axes"


def fig_two_operator_scaling(burgers_path, linear_path, out_path):
    burgers = _load_json(burgers_path)
    linear = _load_json(linear_path)

    fig, ax = plt.subplots()
    for results, label in [(burgers, "burgers"), (linear, "linear")]:
        configs = _sorted_configs(results)
        P = np.array([c["params"] for c in configs], float)
        mean = np.array([np.mean(c["best_list"]) for c in configs])
        std = np.array([np.std(c["best_list"]) for c in configs])
        fit = results["fit"]
        ax.errorbar(
            P, mean, yerr=std, fmt="o-", capsize=3,
            label=rf"{label}: $\alpha$={fit['alpha']:.2f}",
        )

    ref_P = np.array([c["params"] for c in _sorted_configs(burgers)], float)
    ref_mean = np.array([np.mean(c["best_list"]) for c in _sorted_configs(burgers)])
    ref = ref_mean[0] * (ref_P / ref_P[0]) ** (-1.0)
    ax.loglog(ref_P, ref, "k--", alpha=0.6, label=r"benchmark $N^{-1}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters $N$")
    ax.set_ylabel(r"test $H^1$ error")
    ax.set_title("Scaling: burgers vs linear (N=256)")
    ax.legend()
    _save_fig(out_path)
    return "2 operators + benchmark, log-log"


def main():
    apply_paper_style()
    summaries = []

    curves_path = os.path.join(NUMERICS_DIR, "curves_burgers_N256.json")
    results256 = os.path.join(NUMERICS_DIR, "results_burgers_N256.json")
    results64 = os.path.join(NUMERICS_DIR, "results_burgers_N64.json")
    results_linear = os.path.join(NUMERICS_DIR, "results_linear_N256.json")

    # FIG 1
    out1 = os.path.join(REPO_ROOT, "fno_learning_curves_by_size.png")
    if not os.path.isfile(curves_path):
        print(f"ERROR: {curves_path} missing — run "
              "`python run_experiments.py --save_curves` first.", file=sys.stderr)
        sys.exit(1)
    detail1 = fig_learning_curves(curves_path, out1)
    _assert_png(out1)
    summaries.append((out1, f"OK ({detail1})"))

    # FIG 2
    out2 = os.path.join(REPO_ROOT, "fno_instability_vs_size.png")
    detail2 = fig_instability(results256, out2)
    _assert_png(out2)
    summaries.append((out2, f"OK ({detail2})"))

    # FIG 3
    out3 = os.path.join(REPO_ROOT, "fno_best_vs_final.png")
    detail3 = fig_best_vs_final(results256, out3)
    _assert_png(out3)
    summaries.append((out3, f"OK ({detail3})"))

    # FIG 4
    out4 = os.path.join(REPO_ROOT, "fno_alpha_resolution.png")
    detail4 = fig_alpha_resolution(results64, results256, out4)
    _assert_png(out4)
    summaries.append((out4, f"OK ({detail4})"))

    # FIG 5 (conditional)
    out5 = os.path.join(REPO_ROOT, "fno_two_operator_scaling.png")
    if os.path.isfile(results_linear):
        detail5 = fig_two_operator_scaling(results256, results_linear, out5)
        _assert_png(out5)
        summaries.append((out5, f"OK ({detail5})"))
    else:
        print("FIG 5 skipped: results_linear_N256.json not found — run "
              "`python run_experiments.py --operator linear --grid 256 --seeds 5 --epochs 200` "
              "first.")

    # Legacy figures restyled to match paper family
    longrun_path = os.path.join(NUMERICS_DIR, "longrun_curves_N64.json")
    pred_path = os.path.join(NUMERICS_DIR, "prediction_sample_N64.json")

    out_scaling = os.path.join(REPO_ROOT, "scaling_burgers_N256.png")
    detail_s = fig_scaling_burgers(results256, out_scaling)
    _assert_png(out_scaling)
    summaries.append((out_scaling, f"OK ({detail_s})"))

    if os.path.isfile(longrun_path):
        out_lr = os.path.join(REPO_ROOT, "fno_longrun_learning_curves.png")
        detail_lr = fig_longrun(longrun_path, out_lr)
        _assert_png(out_lr)
        summaries.append((out_lr, f"OK ({detail_lr})"))
    else:
        print("fno_longrun_learning_curves.png skipped: run "
              "`python save_legacy_figure_data.py` first.")

    if os.path.isfile(pred_path):
        out_pred = os.path.join(REPO_ROOT, "fno_prediction.png")
        detail_pred = fig_prediction(pred_path, out_pred)
        _assert_png(out_pred)
        summaries.append((out_pred, f"OK ({detail_pred})"))
    else:
        print("fno_prediction.png skipped: run "
              "`python save_legacy_figure_data.py` first.")

    for path, msg in summaries:
        name = os.path.basename(path)
        print(f"{name}: {msg}")


if __name__ == "__main__":
    main()
