"""
Question 5 supplementary experiment: sixth-order compact finite difference scheme.

The scheme used for the first derivative is the classical tridiagonal compact formula

    (1/3) u'_{j-1} + u'_j + (1/3) u'_{j+1}
      = (14/9) (u_{j+1}-u_{j-1})/(2 dx)
        + (1/9) (u_{j+2}-u_{j-2})/(4 dx).

For periodic one-dimensional tests, the derivative is applied through its Fourier symbol,
which is algebraically equivalent to solving the periodic compact linear system.
Time integration is classical RK4, evaluated exactly mode-by-mode for this linear problem.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
FIGDIR = BASE / "figures"
DATADIR = BASE / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)


def compact6_dimless_wavenumber(n: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.fft.fftfreq(n)
    numerator = (14.0 / 9.0) * np.sin(theta) + (1.0 / 18.0) * np.sin(2.0 * theta)
    denominator = 1.0 + (2.0 / 3.0) * np.cos(theta)
    return numerator / denominator


def rk4_linear_advection_compact6(u0: np.ndarray, t_end: float, cfl: float) -> np.ndarray:
    n = len(u0)
    dx = 1.0 / n
    dt_nominal = cfl * dx
    nsteps = int(math.ceil(t_end / dt_nominal))
    dt = t_end / nsteps
    kstar = compact6_dimless_wavenumber(n)
    z = -1j * (kstar / dx) * dt
    amp = (1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0) ** nsteps
    uhat = np.fft.fft(u0)
    u = np.fft.ifft(amp * uhat).real
    return u


def q3_initial(x: np.ndarray, m: int) -> np.ndarray:
    u = np.zeros_like(x)
    for ell in range(1, m + 1):
        u += np.sin(2.0 * np.pi * ell * x)
    return u / m


def l2_error(u: np.ndarray, ue: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u - ue) ** 2)))


def l1_error(u: np.ndarray, ue: np.ndarray) -> float:
    return float(np.mean(np.abs(u - ue)))


def q3_compact6_errors(n: int = 256, ms=(20, 40, 60, 80, 100, 120)) -> pd.DataFrame:
    rows = []
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    for m in ms:
        u0 = q3_initial(x, m)
        u = rk4_linear_advection_compact6(u0, t_end=10.0, cfl=0.3)
        ue = u0.copy()  # t=10 is an integer number of periods
        rows.append({"m": m, "scheme": "Compact6", "L2_error": l2_error(u, ue)})
    return pd.DataFrame(rows)


SEED = 20260420
EPSILON = 0.1
K0 = 24
KMAX = 64
rng = np.random.default_rng(SEED)
PSI = rng.random(KMAX + 1)


def q4_energy_spectrum(k: int, k0: float = K0) -> float:
    return (k / k0) ** 4 * math.exp(-2.0 * (k / k0) ** 2)


def q4_initial(x: np.ndarray) -> np.ndarray:
    u = np.ones_like(x)
    for k in range(1, KMAX + 1):
        u += EPSILON * math.sqrt(q4_energy_spectrum(k)) * np.sin(2.0 * np.pi * k * (x + PSI[k]))
    return u


def q4_compact6_errors(ns=(64, 128, 256, 512, 1024)) -> pd.DataFrame:
    rows = []
    for n in ns:
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        u0 = q4_initial(x)
        u = rk4_linear_advection_compact6(u0, t_end=1.0, cfl=0.2)
        rows.append({"N": n, "scheme": "Compact6", "L1_error": l1_error(u, u0)})
    return pd.DataFrame(rows)


def q4_compact6_resolution(threshold: float = 1e-3) -> tuple[int | None, pd.DataFrame]:
    candidates = list(range(128, 1025, 32))
    rows = []
    first = None
    for n in candidates:
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        u0 = q4_initial(x)
        u = rk4_linear_advection_compact6(u0, t_end=1.0, cfl=0.2)
        err = l1_error(u, u0)
        rows.append({"N": n, "scheme": "Compact6", "L1_error": err})
        if first is None and err <= threshold:
            first = n
    return first, pd.DataFrame(rows)


def plot_q3(q3_new: pd.DataFrame) -> None:
    base = pd.read_csv(DATADIR / "q3_error_results.csv")
    # handle possible column names from previous scripts
    if "L2_error" not in base.columns:
        candidates = [c for c in base.columns if "error" in c.lower()]
        if candidates:
            base = base.rename(columns={candidates[0]: "L2_error"})
    df = pd.concat([base[["m", "scheme", "L2_error"]], q3_new], ignore_index=True)
    order = ["DRP", "DRP-M", "MDCD", "SA-DRP", "Compact6"]
    plt.figure(figsize=(8.0, 5.2))
    for scheme in order:
        sub = df[df["scheme"] == scheme].sort_values("m")
        if len(sub) == 0:
            continue
        plt.semilogy(sub["m"], sub["L2_error"], marker="o", linewidth=1.8, label=scheme)
    plt.xlabel("m")
    plt.ylabel(r"$L_2$ error at $N=256$")
    plt.title("Q3 supplementary comparison with Compact6")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "q5_compact6_q3_extended_m.png", dpi=200)
    plt.close()
    df.to_csv(DATADIR / "q5_compact6_q3_extended_m.csv", index=False)


def plot_q4(q4_new: pd.DataFrame) -> None:
    base = pd.read_csv(DATADIR / "q4_convergence.csv")
    if "L1_error" not in base.columns:
        candidates = [c for c in base.columns if "L1" in c or "error" in c.lower()]
        if candidates:
            base = base.rename(columns={candidates[0]: "L1_error"})
    df = pd.concat([base[["scheme", "N", "L1_error"]], q4_new], ignore_index=True)
    order = ["DRP", "DRP-M", "MDCD", "SA-DRP", "Compact6"]
    plt.figure(figsize=(8.0, 5.2))
    for scheme in order:
        sub = df[df["scheme"] == scheme].sort_values("N")
        if len(sub) == 0:
            continue
        plt.loglog(sub["N"], sub["L1_error"], marker="o", linewidth=1.8, label=scheme)
    plt.axhline(1e-3, linestyle="--", linewidth=1.2, label=r"$10^{-3}$")
    plt.xlabel("N")
    plt.ylabel(r"$L_1$ error")
    plt.title("Q4 supplementary comparison with Compact6")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "q5_compact6_q4_convergence.png", dpi=200)
    plt.close()
    df.to_csv(DATADIR / "q5_compact6_q4_convergence.csv", index=False)


def main() -> None:
    q3_new = q3_compact6_errors()
    q3_new.to_csv(DATADIR / "q5_compact6_q3_only.csv", index=False)
    plot_q3(q3_new)

    q4_new = q4_compact6_errors()
    q4_new.to_csv(DATADIR / "q5_compact6_q4_only.csv", index=False)
    first, hist = q4_compact6_resolution()
    hist.to_csv(DATADIR / "q5_compact6_q4_resolution_history.csv", index=False)
    pd.DataFrame([{"scheme": "Compact6", "min_N_for_L1_le_1e-3": first}]).to_csv(
        DATADIR / "q5_compact6_q4_resolution_threshold.csv", index=False
    )
    plot_q4(q4_new)

    print("Q3 Compact6 errors")
    print(q3_new.to_string(index=False))
    print("\nQ4 Compact6 errors")
    print(q4_new.to_string(index=False))
    print("\nQ4 Compact6 threshold N", first)


if __name__ == "__main__":
    main()
