"""
parameter_sweep.py

Parameter sweep experiment to study FNO approximation rates.
Tests different model sizes and plots error vs. parameter count.
"""

import numpy as np
import matplotlib.pyplot as plt
from train_fno import (
    generate_dataset,
    train_fno,
    count_parameters,
    FNO1d,
)


def run_parameter_sweep():
    """Run experiments with different FNO configurations."""
    
    # PDE parameters
    N = 64  # Reduced for stability
    nu = 0.01
    L = 1.0
    t_final = 1.0
    dt = 2e-3  # Larger timestep for speed
    h1_radius = 0.3  # Smaller radius for stability

    # Generate dataset once
    print("Generating dataset for parameter sweep...")
    x_train, y_train, x_test, y_test = generate_dataset(
        n_train=256,  # Reduced for speed
        n_test=64,
        N=N,
        nu=nu,
        L=L,
        t_final=t_final,
        dt=dt,
        h1_radius=h1_radius,
        seed=42,
    )

    # Model configurations to test (reduced for faster testing)
    configs = [
        {"modes": 8, "width": 32},
        {"modes": 12, "width": 48},
        {"modes": 16, "width": 64},
    ]

    results = []

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 60)
        print(f"Experiment {i+1}/{len(configs)}: modes={cfg['modes']}, width={cfg['width']}")
        print("=" * 60)

        model, test_loss = train_fno(
            x_train,
            y_train,
            x_test,
            y_test,
            modes=cfg["modes"],
            width=cfg["width"],
            batch_size=32,
            lr=1e-3,
            n_epochs=30,  # Reduced for speed
            L=L,
        )

        # Count parameters
        n_params = count_parameters(model)
        results.append({"params": n_params, "error": test_loss, "config": cfg})
        print(f"Results: {n_params:,} params, H¹-error = {test_loss:.4e}")

    return results


def plot_results(results):
    """Plot approximation error vs. parameter count."""
    params = np.array([r["params"] for r in results])
    errors = np.array([r["error"] for r in results])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(params, errors, "o-", linewidth=2, markersize=8, label="FNO")

    # Fit power law
    log_p = np.log(params)
    log_e = np.log(errors)
    coef = np.polyfit(log_p, log_e, 1)
    slope = coef[0]

    # Plot fitted line
    p_fit = np.linspace(params.min(), params.max(), 100)
    e_fit = np.exp(coef[1]) * p_fit ** slope
    ax.loglog(
        p_fit, e_fit, "--", color="gray", alpha=0.7, label=f"Rate: {slope:.2f}"
    )

    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Test H¹-Error", fontsize=12)
    ax.set_title(
        "FNO Approximation of Burgers Solution Operator", fontsize=14, fontweight="bold"
    )
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("parameter_sweep_results.png", dpi=150)
    print("\nSaved plot to parameter_sweep_results.png")
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Empirical approximation rate: error ~ N^{slope:.2f}")
    print("\nDetailed results:")
    for r in results:
        cfg = r["config"]
        print(
            f"  modes={cfg['modes']:2d}, width={cfg['width']:3d} → "
            f"{r['params']:7,} params, error={r['error']:.4e}"
        )


if __name__ == "__main__":
    results = run_parameter_sweep()
    plot_results(results)
