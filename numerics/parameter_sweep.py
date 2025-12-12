"""
Basic training script for Fourier Neural Operator on 1D Burgers' equation.
Includes:
- model-size sweep
- global relative H^1 error
- multiple plots (error vs params, learning curves, long runs)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# Spectral solver for 1D viscous Burgers' equation
# ============================================================

def spectral_derivative(u, L=1.0, order=1):
    """
    Compute spectral derivative of u(x) on [0, L] with periodic BC.
    u: array of shape (N,) (real-valued)
    order: 1 or 2
    """
    N = u.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    u_hat = np.fft.fft(u)
    if order == 1:
        du_hat = 1j * k * u_hat
    elif order == 2:
        du_hat = -(k**2) * u_hat
    else:
        raise ValueError("Only first and second derivatives are implemented.")
    du = np.fft.ifft(du_hat)
    return np.real(du)


def burgers_spectral_solver(u0, nu=0.01, t_final=1.0, dt=1e-3, L=1.0):
    """
    Spectral solver for 1D viscous Burgers using integrating factor + RK2.
    Treats diffusion exactly in Fourier space.
    """
    N = u0.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)

    # Integrating factor for diffusion: exp(-nu * k^2 * dt/2) for half-step
    exp_factor_half = np.exp(-nu * (k**2) * dt / 2.0)

    u_hat = np.fft.fft(u0)
    n_steps = int(t_final / dt)

    for _ in range(n_steps):
        # Half step with diffusion
        u_hat = exp_factor_half * u_hat
        u = np.real(np.fft.ifft(u_hat))

        # Nonlinear term at half step
        u_x = spectral_derivative(u, L=L, order=1)
        nonlinear = -u * u_x
        nl_hat = np.fft.fft(nonlinear)

        # Full step with nonlinearity
        u_hat = exp_factor_half * (u_hat + dt * nl_hat)
        u = np.real(np.fft.ifft(u_hat))

        # Safety check
        if not np.isfinite(u).all() or np.abs(u).max() > 1e3:
            return np.full_like(u0, np.nan)

        u_hat = np.fft.fft(u)

    return u


# ============================================================
# Data generation
# ============================================================

def h1_norm(u, L=1.0):
    """Discrete H^1 norm (numpy version, for IC scaling)."""
    u_x = spectral_derivative(u, L=L, order=1)
    integrand = u**2 + u_x**2
    return np.sqrt(np.mean(integrand))


def sample_smooth_ic(N, L=1.0, h1_radius=0.5, rng=None):
    """Sample a random smooth initial condition with bounded H^1 norm."""
    if rng is None:
        rng = np.random.default_rng()

    u = rng.normal(loc=0.0, scale=1.0, size=N)
    k = np.fft.fftfreq(N, d=L / N)
    u_hat = np.fft.fft(u)

    # Gaussian low-pass filter
    alpha = 12.0
    filter_ = np.exp(-alpha * (k**2))
    u_hat_filtered = u_hat * filter_
    u_smooth = np.real(np.fft.ifft(u_hat_filtered))

    u_smooth -= np.mean(u_smooth)

    # Scale to H^1 radius
    norm = h1_norm(u_smooth, L=L)
    if norm > 1e-8:
        u_smooth = (h1_radius / norm) * u_smooth

    return u_smooth


def generate_dataset(
    n_train=512,
    n_test=128,
    N=128,
    nu=0.01,
    L=1.0,
    t_final=1.0,
    dt=1e-3,
    h1_radius=0.5,
    seed=42,
):
    """Generate (u0, u(·,1)) pairs for training and testing."""
    rng = np.random.default_rng(seed)
    print(f"Generating dataset (H^1-ball, radius={h1_radius})...")

    def generate_split(n_samples):
        u0_list = []
        uT_list = []
        i = 0
        attempts = 0
        max_attempts = n_samples * 3

        while i < n_samples and attempts < max_attempts:
            attempts += 1
            u0 = sample_smooth_ic(N, L=L, h1_radius=h1_radius, rng=rng)
            uT = burgers_spectral_solver(u0, nu=nu, t_final=t_final, dt=dt, L=L)

            if np.isfinite(u0).all() and np.isfinite(uT).all():
                u0_list.append(u0.astype(np.float32))
                uT_list.append(uT.astype(np.float32))
                i += 1
                if i % max(1, n_samples // 10) == 0:
                    print(f"  generated {i}/{n_samples}")

        if i < n_samples:
            print(f"Warning: Could only generate {i}/{n_samples} stable samples")
        return np.stack(u0_list, axis=0), np.stack(uT_list, axis=0)

    x_train, y_train = generate_split(n_train)
    x_test, y_test = generate_split(n_test)

    print("Done generating dataset.")
    return x_train, y_train, x_test, y_test


# ============================================================
# PyTorch components
# ============================================================

class BurgersDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def h1_loss_torch(u_pred, u_true, L=1.0):
    """
    Discrete H^1 loss using finite differences (NO sqrt).
    Returns:
      loss = mean( (u_pred-u_true)^2 + (ux_pred-ux_true)^2 )
      l2_term, grad_term (detached) for logging if needed.
    """
    dx = L / u_pred.shape[1]
    diff = u_pred - u_true
    l2_term = torch.mean(diff**2)

    def grad(u):
        return (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2.0 * dx)

    g_pred = grad(u_pred)
    g_true = grad(u_true)
    grad_term = torch.mean((g_pred - g_true) ** 2)

    return l2_term + grad_term, l2_term.detach(), grad_term.detach()


def compute_global_relative_h1(model, x, y, L=1.0, device="cpu", batch_size=64):
    """
    Compute global absolute and relative H^1 error over a dataset.

    H^1(u) ~ sqrt(mean_x (u^2 + (ux)^2))
    relative = H^1(u_pred - u_true) / H^1(u_true)
    """
    ds = BurgersDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    dx = L / x.shape[1]

    def grad(u):
        return (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2.0 * dx)

    model.eval()
    sum_err = 0.0
    sum_true = 0.0
    count = 0

    with torch.no_grad():
        for u0_batch, uT_batch in loader:
            u0_batch = u0_batch.to(device)
            uT_batch = uT_batch.to(device)

            pred = model(u0_batch)

            g_true = grad(uT_batch)
            g_pred = grad(pred)

            err = pred - uT_batch
            err_g = g_pred - g_true

            integrand_err = err**2 + err_g**2
            integrand_true = uT_batch**2 + g_true**2

            sum_err += integrand_err.sum().item()
            sum_true += integrand_true.sum().item()
            count += integrand_err.numel()

    mean_err = sum_err / count
    mean_true = sum_true / count

    H1_error = np.sqrt(mean_err)
    H1_true = np.sqrt(mean_true)
    rel = H1_error / (H1_true + 1e-12)

    return H1_error, rel


# ============================================================
# FNO architecture
# ============================================================

class SpectralConv1d(nn.Module):
    """1D Fourier convolution layer."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes, 2)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bim, iom -> bom", input, weights)

    def forward(self, x):
        batchsize, in_channels, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        N_rfft = x_ft.shape[-1]
        out_ft = torch.zeros(
            batchsize, self.out_channels, N_rfft, device=x.device, dtype=torch.cfloat
        )

        w = torch.view_as_complex(self.weights)
        modes = min(self.modes, N_rfft)
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], w[:, :, :modes])

        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x_out


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1D inputs."""

    def __init__(self, modes=16, width=64):
        super().__init__()
        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(2, width)
        self.conv_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(4)]
        )
        self.w_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, u0):
        batchsize, N = u0.shape
        device = u0.device

        x_coord = torch.linspace(0, 1, N, device=device).unsqueeze(0)
        x_coord = x_coord.repeat(batchsize, 1)

        inp = torch.stack([u0, x_coord], dim=-1)
        x = self.fc0(inp)
        x = x.permute(0, 2, 1)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        out = x.squeeze(-1)
        return out


# ============================================================
# Training
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fno(
    x_train,
    y_train,
    x_test,
    y_test,
    modes=16,
    width=64,
    batch_size=32,
    lr=1e-3,
    n_epochs=100,
    L=1.0,
    device=None,
    log_every=10,
    return_history=False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = BurgersDataset(x_train, y_train)
    test_ds = BurgersDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = FNO1d(modes=modes, width=width).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "test": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        for u0_batch, uT_batch in train_loader:
            u0_batch = u0_batch.to(device)
            uT_batch = uT_batch.to(device)

            optimizer.zero_grad()
            pred = model(u0_batch)
            loss, _, _ = h1_loss_torch(pred, uT_batch, L=L)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * u0_batch.size(0)

        train_loss = train_loss_accum / len(train_ds)

        # evaluate test loss every epoch
        model.eval()
        test_loss_accum = 0.0
        with torch.no_grad():
            for u0_batch, uT_batch in test_loader:
                u0_batch = u0_batch.to(device)
                uT_batch = uT_batch.to(device)
                pred = model(u0_batch)
                loss, _, _ = h1_loss_torch(pred, uT_batch, L=L)
                test_loss_accum += loss.item() * u0_batch.size(0)
        test_loss = test_loss_accum / len(test_ds)

        history["train"].append(train_loss)
        history["test"].append(test_loss)

        if (epoch % log_every == 0) or (epoch == 1):
            print(
                f"Epoch {epoch:4d}: "
                f"train H1-loss = {train_loss:.4e}, "
                f"test H1-loss = {test_loss:.4e}"
            )

    final_test_loss = history["test"][-1]
    print(f"Final test H1-loss: {final_test_loss:.4e}")

    if return_history:
        return model, final_test_loss, history
    else:
        return model, final_test_loss


# ============================================================
# Visualization
# ============================================================

def plot_example(model, x_test, y_test, idx=0, L=1.0, device="cpu"):
    """Plot prediction vs true solution for a test sample."""
    model.eval()
    u0 = torch.from_numpy(x_test[idx : idx + 1]).float().to(device)
    uT_true = torch.from_numpy(y_test[idx : idx + 1]).float().to(device)

    with torch.no_grad():
        uT_pred = model(u0)

    u0 = u0.cpu().numpy().squeeze()
    uT_true = uT_true.cpu().numpy().squeeze()
    uT_pred = uT_pred.cpu().numpy().squeeze()

    N = u0.shape[0]
    x = np.linspace(0, L, N, endpoint=False)

    # Finite-diff gradients
    dx = L / N

    def grad(u):
        return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)

    g_true = grad(uT_true)
    g_pred = grad(uT_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, u0, label="u(x,0)", alpha=0.7)
    ax1.plot(x, uT_true, label="u(x,1) true", linewidth=2)
    ax1.plot(x, uT_pred, "--", label="u(x,1) pred", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.set_title("Solution at t=1")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, g_true, label="∂ₓu true", linewidth=2)
    ax2.plot(x, g_pred, "--", label="∂ₓu pred", linewidth=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("∂ₓu")
    ax2.set_title("Derivative at t=1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fno_prediction.png", dpi=150)
    print("Saved visualization to fno_prediction.png")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # PDE parameters
    N = 64  # spatial grid points
    nu = 0.01
    L = 1.0
    t_final = 1.0
    dt = 2e-3

    # Dataset size
    n_train = 256
    n_test = 64
    h1_radius = 0.3  # small for stability

    # Generate ONE dataset to reuse across all model sizes
    x_train, y_train, x_test, y_test = generate_dataset(
        n_train=n_train,
        n_test=n_test,
        N=N,
        nu=nu,
        L=L,
        t_final=t_final,
        dt=dt,
        h1_radius=h1_radius,
        seed=42,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Model-size sweep: list of (modes, width) configurations
    # --------------------------------------------------------
    configs = [
        {"modes": 8,  "width": 32},
        {"modes": 12, "width": 48},
        {"modes": 16, "width": 64},
        {"modes": 24, "width": 96},
    ]

    sweep_epochs = 100
    results = []      # (num_params, test_H1_loss, rel_H1)
    histories = {}    # key: (modes, width) -> history dict

    for cfg in configs:
        print("=" * 60)
        print(
            f"Training FNO with modes={cfg['modes']}, "
            f"width={cfg['width']}, epochs={sweep_epochs}"
        )

        model, test_loss, history = train_fno(
            x_train,
            y_train,
            x_test,
            y_test,
            modes=cfg["modes"],
            width=cfg["width"],
            batch_size=32,
            lr=1e-3,
            n_epochs=sweep_epochs,
            L=L,
            device=device,
            log_every=10,
            return_history=True,
        )

        n_params = count_parameters(model)
        H1_abs, rel_h1 = compute_global_relative_h1(
            model, x_test, y_test, L=L, device=device, batch_size=64
        )
        print(
            f"Global absolute H1 error: {H1_abs:.4e}, "
            f"Global relative H1 error: {rel_h1:.3e}"
        )

        results.append((n_params, test_loss, rel_h1))
        histories[(cfg["modes"], cfg["width"])] = history

        print(
            f"Params: {n_params}, Test H1-loss: {test_loss:.4e}, "
            f"Rel H1: {rel_h1:.3e}"
        )

    print("\nAll (params, test_H1_loss, rel_H1) triples:")
    for n_params, err, rel in results:
        print(f"N = {n_params:8d},  H1-error = {err:.4e},  rel_H1 = {rel:.3e}")

    # --------------------------------------------------------
    # Log–log plot of error vs model size (absolute H1-loss)
    # --------------------------------------------------------
    params = np.array([p for p, e, r in results], dtype=float)
    errors = np.array([e for p, e, r in results], dtype=float)
    rel_errors = np.array([r for p, e, r in results], dtype=float)

    plt.figure()
    plt.loglog(params, errors, "o-")
    plt.xlabel("Number of parameters $N$")
    plt.ylabel("Test $H^1$-loss")
    plt.title("FNO approximation of Burgers solution operator")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("figure_1.png", dpi=150)
    print("Saved log–log plot to figure_1.png")

    # Empirical slope (power-law exponent) using absolute H1-loss
    log_p = np.log(params)
    log_e = np.log(errors)
    slope, intercept = np.polyfit(log_p, log_e, 1)
    print(f"Empirical exponent alpha ≈ {-slope:.3f}")

    # --------------------------------------------------------
    # Error-vs-params: absolute + relative
    # --------------------------------------------------------
    plt.figure()
    plt.loglog(params, errors, "o-", label=r"$\|G-G_\theta\|_{H^1}$ (loss)")
    plt.loglog(params, rel_errors, "s--", label=r"relative $H^1$ error")
    plt.xlabel("Number of parameters $N$")
    plt.ylabel("Error")
    plt.title("Burgers solution operator: error vs model size")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fno_error_vs_params.png", dpi=150)
    print("Saved error-vs-params plot to fno_error_vs_params.png")

    # --------------------------------------------------------
    # Learning curves: test loss vs epoch for each model size
    # --------------------------------------------------------
    plt.figure()
    for (modes, width), history in histories.items():
        epochs = np.arange(1, len(history["test"]) + 1)
        plt.semilogy(
            epochs,
            history["test"],
            label=f"modes={modes}, width={width}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test $H^1$-loss")
    plt.title("Learning curves for different FNO sizes")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fno_learning_curves_by_size.png", dpi=150)
    print("Saved learning-curve plot to fno_learning_curves_by_size.png")

    # --------------------------------------------------------
    # Long runs for the largest model (modes=24, width=96)
    # --------------------------------------------------------
    big_cfg = configs[-1]
    long_epochs_list = [100, 500, 1000]
    long_histories = {}
    long_models = {}

    for n_ep in long_epochs_list:
        print("=" * 60)
        print(
            f"Long run: modes={big_cfg['modes']}, width={big_cfg['width']}, "
            f"epochs={n_ep}"
        )
        model_long, loss_long, hist_long = train_fno(
            x_train,
            y_train,
            x_test,
            y_test,
            modes=big_cfg["modes"],
            width=big_cfg["width"],
            batch_size=32,
            lr=1e-3,
            n_epochs=n_ep,
            L=L,
            device=device,
            log_every=10,
            return_history=True,
        )
        H1_abs_long, rel_long = compute_global_relative_h1(
            model_long, x_test, y_test, L=L, device=device, batch_size=64
        )
        print(
            f"[Long run] epochs={n_ep}, test H1-loss={loss_long:.4e}, "
            f"rel_H1={rel_long:.3e}"
        )

        long_histories[n_ep] = hist_long
        long_models[n_ep] = model_long

    # Long-run learning curves (test loss vs epoch)
    plt.figure()
    for n_ep, hist in long_histories.items():
        epochs = np.arange(1, len(hist["test"]) + 1)
        plt.semilogy(epochs, hist["test"], label=f"{n_ep} epochs")

    plt.xlabel("Epoch")
    plt.ylabel("Test $H^1$-loss")
    plt.title("Long-run learning curves (modes=24, width=96)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fno_longrun_learning_curves.png", dpi=150)
    print("Saved long-run learning curves to fno_longrun_learning_curves.png")

    # --------------------------------------------------------
    # Qualitative visualization using the 1000-epoch model
    # --------------------------------------------------------
    model_final = long_models[max(long_models.keys())]
    plot_example(model_final, x_test, y_test, idx=0, L=L, device=device)

    # Save the final (largest + longest-trained) model
    torch.save(model_final.state_dict(), "fno_burgers_largest.pt")
    print("Saved largest model to fno_burgers_largest.pt")
