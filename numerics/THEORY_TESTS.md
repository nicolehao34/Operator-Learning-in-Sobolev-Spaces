# Theory-test experiments

Empirical probes of the scaling law **error ~ N^(-s/d)**. Each command below is
self-contained; full sweeps are long — use `--quick` only for smoke tests.

## Benchmark exponents

| Test | Operator | d | s (default) | benchmark s/d |
|------|----------|---|-------------|---------------|
| (a) | `burgers` | 1 | 1 | **1.0** |
| (b) | `linear` | 1 | 1 | **1.0** |
| (c) | `adv_diff_2d` | 2 | 1 | **0.5** |
| (d) | `burgers` + `--ic_smoothness` | 1 | varies | **s/1 = s** |

Results JSON includes `{operator, d, s, benchmark_exponent, configs, fit, total_n_spiked}`.

## Commands

Run from repo root:

```bash
cd numerics
source ../.venv/bin/activate   # Python 3.10 + requirements.txt
```

### (a) 1D Burgers — benchmark s/d = 1

```bash
python run_theory_tests.py --operator burgers --grid 256 --seeds 5 --epochs 200
```

Smoke:

```bash
python run_theory_tests.py --operator burgers --quick
```

### (b) 1D linear advection–diffusion — benchmark s/d = 1

```bash
python run_theory_tests.py --operator linear --grid 256 --seeds 5 --epochs 200
```

Smoke:

```bash
python run_theory_tests.py --operator linear --quick
```

### (c) 2D linear advection–diffusion — benchmark s/d = 1/2

```bash
python run_theory_tests.py --operator adv_diff_2d --grid 64 --seeds 5 --epochs 200
```

Smoke:

```bash
python run_theory_tests.py --operator adv_diff_2d --quick
```

### (d) Variable input smoothness s (1D Burgers)

Sweep `s` and check whether fitted α tracks s/d:

```bash
python run_theory_tests.py --operator burgers --ic_smoothness 0.5 --grid 256 --seeds 5 --epochs 200
python run_theory_tests.py --operator burgers --ic_smoothness 1.0 --grid 256 --seeds 5 --epochs 200
python run_theory_tests.py --operator burgers --ic_smoothness 2.0 --grid 256 --seeds 5 --epochs 200
```

Smoke (prints IC spectrum diagnostic):

```bash
python run_theory_tests.py --operator burgers --ic_smoothness 2.0 --quick
```

**IC knob mapping:** `--ic_smoothness s` applies a Fourier filter |k|^{-s} before
H^1 normalization. The smoke-test prints an estimated effective decay exponent
from one sample (should increase with s).

### (e) Optimization-stability ablation (grad clip + cosine LR)

Rerun largest 1D models with stabilization; compare `total_n_spiked` in JSON:

```bash
python run_experiments.py --operator burgers --grid 256 --seeds 5 --epochs 200 \
  --grad_clip 1.0 --lr_schedule cosine
```

Smoke:

```bash
python run_experiments.py --operator burgers --quick --grad_clip 1.0 --lr_schedule cosine
```

## Multi-operator overlay figure

After running (a) and (b) at the same grid:

```bash
python make_paper_figures.py --overlay results_burgers_N256.json results_linear_N256.json
```

Or from repo root:

```bash
python numerics/make_paper_figures.py --overlay \
  numerics/results_burgers_N256.json numerics/results_linear_N256.json
```

Writes `fno_two_operator_scaling.png` with per-operator α and benchmark lines.

## Outputs

- `results_<operator>_N<grid>.json` — per-config `best_list`, `final_list`, `n_spiked`
- `scaling_<operator>_N<grid>_<YYYYMMDD_HHMMSS>.png` — log-log plot (timestamped;
  canonical `scaling_*.png` files are never overwritten)

Paper figures from `make_paper_figures.py` follow the same rule: e.g.
`fno_prediction_20260624_233045.png` alongside the original `fno_prediction.png`.
