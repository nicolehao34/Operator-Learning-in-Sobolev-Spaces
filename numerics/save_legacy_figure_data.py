"""
Train and cache data for legacy paper figures (long-run curves, prediction panel).

Run once; make_paper_figures.py reads the JSON outputs.
"""

import json
import os

import numpy as np
import torch

from run_experiments import make_dataset, train_one
from train_fno import spectral_derivative

NUMERICS_DIR = os.path.dirname(os.path.abspath(__file__))
LONGRUN_PATH = os.path.join(NUMERICS_DIR, "longrun_curves_N64.json")
PRED_PATH = os.path.join(NUMERICS_DIR, "prediction_sample_N64.json")


def _prediction_arrays(model, x_test, y_test, idx, L, device):
    model.eval()
    u0_t = torch.from_numpy(x_test[idx : idx + 1]).float().to(device)
    with torch.no_grad():
        uT_pred = model(u0_t).cpu().numpy().squeeze()
    u0 = x_test[idx]
    uT_true = y_test[idx]
    N = u0.shape[0]
    x = np.linspace(0, L, N, endpoint=False)
    g_true = spectral_derivative(uT_true, L=L, order=1)
    g_pred = spectral_derivative(uT_pred, L=L, order=1)
    return {
        "L": L, "idx": idx,
        "x": x.tolist(),
        "u0": u0.tolist(),
        "uT_true": uT_true.tolist(),
        "uT_pred": uT_pred.tolist(),
        "grad_true": g_true.tolist(),
        "grad_pred": g_pred.tolist(),
    }


def main():
    L = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    xtr, ytr, xte, yte = make_dataset(
        256, 64, 64, "burgers", data_seed=42,
        nu=0.01, t_final=1.0, dt=2e-3, L=L,
    )
    print(f"dataset: train={xtr.shape} test={xte.shape}")

    modes, width = 24, 96
    long_runs = []
    model = None
    for n_ep in [100, 500, 1000]:
        print(f"=== long run: {n_ep} epochs ===")
        r = train_one(
            xtr, ytr, xte, yte, modes, width, n_ep, 1e-3, 32, L, device,
            model_seed=42, eval_every=1,
            return_model=(n_ep == 1000),
        )
        if n_ep == 1000:
            model = r.pop("model")
        long_runs.append({
            "max_epochs": n_ep,
            "modes": modes,
            "width": width,
            "params": r["params"],
            "epochs": r["epochs"],
            "test_loss": r["test_loss"],
        })

    with open(LONGRUN_PATH, "w") as f:
        json.dump({"grid": 64, "runs": long_runs}, f, indent=2)
    print(f"wrote {LONGRUN_PATH}")

    pred = _prediction_arrays(model, xte, yte, idx=0, L=L, device=device)
    pred.update({"modes": modes, "width": width, "train_epochs": 1000})
    with open(PRED_PATH, "w") as f:
        json.dump(pred, f, indent=2)
    print(f"wrote {PRED_PATH}")


if __name__ == "__main__":
    main()
