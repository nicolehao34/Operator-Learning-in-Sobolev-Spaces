"""
Model-size sweep to estimate empirical alpha using best-epoch errors.

- Generates a single dataset (H^1-ball initial conditions).
- Trains several FNO models with different (modes, width) for `sweep_epochs` epochs.
- Keeps full test-loss history for each configuration.
- Computes empirical exponent alpha from a power law fit:
      ||G - G_θ||_{H^1} ≈ C N^{-alpha}
  using the *best* test loss (minimum over epochs) per model.
- Produces:
  - log log plot of best-epoch error vs parameter count
  - learning curves for each model size
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
    N = u0.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)

    exp_factor_half = np.exp(-nu * (k**2) * dt / 2.0)
    u_hat = np.fft.fft(u0)
    n_steps = int(t_final / dt)

    for _ in range(n_steps):
        u_hat = exp_factor_half * u_hat
        u = np.real(np.fft.ifft(u_hat))

        u_x = spectral_derivative(u, L=L, order=1)
        nonlinear = -u * u_x
        nl_hat = np.fft.fft(nonlinear)

        u_hat = exp_factor_half * (u_hat + dt * nl_hat)
        u = np.real(np.fft.ifft(u_hat))

        if not np.isfinite(u).all() or np.abs(u).max() > 1e3:
            return np.full_like(u0, np.nan)

        u_hat = np.fft.fft(u)

    return u


# ============================================================
# Data generation
# ============================================================

def h1_norm(u, L=1.0):
    u_x = spectral_derivative(u, L=L, order=1)
    integrand = u**2 + u_x**2
    return np.sqrt(np.mean(integrand))


def sample_smooth_ic(N, L=1.0, h1_radius=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    u = rng.normal(loc=0.0, scale=1.0, size=N)
    k = np.fft.fftfreq(N, d=L / N)
    u_hat = np.fft.fft(u)

    alpha = 12.0
    filter_ = np.exp(-alpha * (k**2))
    u_hat_filtered = u_hat * filter_
    u_smooth = np.real(np.fft.ifft(u_hat_filtered))

    u_smooth -= np.mean(u_smooth)
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
    rng = np.random.default_rng(seed)
    print(f"Generating dataset (H^1-ball, radius={h1_radius})...")

    def generate_split(n_samples):
        u0_list, uT_list = [], []
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
    dx = L / u_pred.shape[1]
    diff = u_pred - u_true
    l2_term = torch.mean(diff**2)

    def grad(u):
        return (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2.0 * dx)

    g_pred = grad(u_pred)
    g_true = grad(u_true)
    grad_term = torch.mean((g_pred - g_true) ** 2)
    return l2_term + grad_term, l2_term.detach(), grad_term.detach()


# ============================================================
# FNO architecture
# ============================================================

class SpectralConv1d(nn.Module):
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
    n_epochs=500,
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
    n_params = count_parameters(model)
    print(f"Model (modes={modes}, width={width}) has {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "test": [], "n_params": n_params}

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
    print(
        f"Final test H1-loss (modes={modes}, width={width}): "
        f"{final_test_loss:.4e}"
    )

    if return_history:
        return model, final_test_loss, history
    else:
        return model, final_test_loss


# ============================================================
# Main – size sweep + best-epoch alpha
# ============================================================

if __name__ == "__main__":
    # PDE / data parameters
    N = 64
    nu = 0.01
    L = 1.0
    t_final = 1.0
    dt = 2e-3

    n_train = 256
    n_test = 64
    h1_radius = 0.3

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

    # Model-size configs
    configs = [
        {"modes": 8,  "width": 32},
        {"modes": 12, "width": 48},
        {"modes": 16, "width": 64},
        {"modes": 24, "width": 96},
    ]

    sweep_epochs = 500   # longer run to get closer to approximation regime
    histories = {}
    final_results = []   # (n_params, final_test_loss)

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
        histories[(cfg["modes"], cfg["width"])] = history
        final_results.append((history["n_params"], test_loss))

    print("\nFinal-epoch results (for reference):")
    for n_params, err in final_results:
        print(f"N = {n_params:8d}, final test H1-loss = {err:.4e}")

    # --------------------------------------------------------
    # Compute best-epoch errors and empirical alpha
    # --------------------------------------------------------
    params = []
    best_errors = []

    for cfg in configs:
        key = (cfg["modes"], cfg["width"])
        history = histories[key]
        n_params = history["n_params"]
        best_test_loss = min(history["test"])
        params.append(n_params)
        best_errors.append(best_test_loss)
        print(
            f"Config modes={cfg['modes']}, width={cfg['width']}: "
            f"N={n_params}, best test H1-loss={best_test_loss:.4e}"
        )

    params = np.array(params, dtype=float)
    best_errors = np.array(best_errors, dtype=float)

    # Fit power law in log–log space: log(err) = a * log(N) + b
    log_p = np.log(params)
    log_e = np.log(best_errors)
    slope, intercept = np.polyfit(log_p, log_e, 1)
    alpha = -slope
    C = np.exp(intercept)
    print(f"\nEmpirical exponent alpha (best-epoch) ≈ {alpha:.3f}")
    print(f"Fitted law: ||G - G_θ||_{'{H^1}'} ≈ {C:.3e} * N^(-{alpha:.3f})")

    # --------------------------------------------------------
    # Plot: best-epoch error vs parameters 
    # --------------------------------------------------------
    plt.figure()
    plt.loglog(params, best_errors, "o-", label="best-epoch test $H^1$-loss")

    # also plot fitted line for visualization
    N_line = np.linspace(params.min(), params.max(), 100)
    err_line = C * N_line ** (-alpha)
    plt.loglog(N_line, err_line, "--", label=f"fit: $CN^{{-{alpha:.2f}}}$")

    plt.xlabel("Number of parameters $N$")
    plt.ylabel("Best-epoch test $H^1$-loss")
    plt.title("FNO approximation of Burgers solution operator (best epoch)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fno_best_epoch_error_vs_params.png", dpi=150)
    print("Saved best-epoch error vs params plot to fno_best_epoch_error_vs_params.png")

    # --------------------------------------------------------
    # Learning curves for each model size
    # --------------------------------------------------------
    plt.figure()
    for cfg in configs:
        key = (cfg["modes"], cfg["width"])
        history = histories[key]
        epochs = np.arange(1, len(history["test"]) + 1)
        plt.semilogy(
            epochs,
            history["test"],
            label=f"modes={cfg['modes']}, width={cfg['width']}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test $H^1$-loss")
    plt.title("Learning curves for different FNO sizes")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fno_learning_curves_by_size_best_epoch_sweep.png", dpi=150)
    print("Saved learning curves to fno_learning_curves_by_size_best_epoch_sweep.png")
