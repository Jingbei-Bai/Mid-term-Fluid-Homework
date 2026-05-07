"""
Clean rebuild for Problem 3.

This script implements the one-dimensional wave-packet test in
Li & Ren (2023), Physics of Fluids 35, 036114, Sec. III A.
It does not depend on the previous Problem 3 code.

Equation:
    u_t + u_x = 0, x in [0,1], periodic boundary conditions.
Initial condition:
    u(x,0) = (1/m) sum_{l=1}^m sin(2*pi*l*x).
Time integration:
    classical four-stage Runge-Kutta, CFL=0.3.
Schemes:
    DRP, DRP-M, MDCD, SA-DRP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = 1.0
CFL = 0.3
T_END = 10.0
EPS_SENSOR = 1.0e-8
SMALL_AMPLITUDE_THRESHOLD = 1.0e-3

# Convergence points are chosen to match the locations shown in the paper figure.
PAPER_CONVERGENCE_N = [50, 100, 200, 400, 1000]
CONVERGENCE_M = [5, 10, 15, 20]
EXTENDED_M = [20, 40, 60, 80, 100, 120]

SCHEME_ORDER = ["DRP", "DRP-M", "MDCD", "SA-DRP"]


@dataclass(frozen=True)
class RunResult:
    x: np.ndarray
    u: np.ndarray
    exact: np.ndarray
    l2: float
    dt: float
    nsteps: int


def periodic_shift(values: np.ndarray, offset: int) -> np.ndarray:
    """Return values[j+offset] for periodic data stored at index j."""
    return np.roll(values, -offset)


def wave_packet(x: np.ndarray, m: int) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)
    for ell in range(1, m + 1):
        result += np.sin(2.0 * math.pi * ell * x)
    return result / float(m)


def exact_solution(x: np.ndarray, t: float, m: int) -> np.ndarray:
    return wave_packet((x - A * t) % 1.0, m)


def l2_error(numerical: np.ndarray, exact: np.ndarray) -> float:
    return float(np.sqrt(np.mean((numerical - exact) ** 2)))


# -----------------------------------------------------------------------------
# DRP and DRP-M explicit derivative forms
# -----------------------------------------------------------------------------

def seven_point_central_derivative(u: np.ndarray, dx: float, a1: float, a2: float, a3: float) -> np.ndarray:
    return (
        a1 * (periodic_shift(u, 1) - periodic_shift(u, -1))
        + a2 * (periodic_shift(u, 2) - periodic_shift(u, -2))
        + a3 * (periodic_shift(u, 3) - periodic_shift(u, -3))
    ) / dx


def rhs_drp(u: np.ndarray, dx: float) -> np.ndarray:
    # Tam and Webb DRP coefficients.
    return -seven_point_central_derivative(u, dx, 0.79926643, -0.18941314, 0.02651995)


def rhs_drp_m(u: np.ndarray, dx: float) -> np.ndarray:
    # Modified DRP coefficients used by Li and Ren (2023).
    return -seven_point_central_derivative(u, dx, 0.770882380518, -0.166705904415, 0.020843142770)


# -----------------------------------------------------------------------------
# MDCD flux-difference form
# -----------------------------------------------------------------------------

def six_point_interface_flux(u: np.ndarray, cdisp: np.ndarray, cdiss: np.ndarray) -> np.ndarray:
    """Compute \hat{f}_{j+1/2} for the six-point MDCD/SA-DRP stencil.

    The returned array is indexed by j and represents the interface j+1/2.
    For linear advection with a=1, f_j = u_j.
    """
    ujm2 = periodic_shift(u, -2)
    ujm1 = periodic_shift(u, -1)
    uj = u
    ujp1 = periodic_shift(u, 1)
    ujp2 = periodic_shift(u, 2)
    ujp3 = periodic_shift(u, 3)
    return (
        (0.5 * cdisp + 0.5 * cdiss) * ujm2
        + (-1.5 * cdisp - 2.5 * cdiss - 1.0 / 12.0) * ujm1
        + (cdisp + 5.0 * cdiss + 7.0 / 12.0) * uj
        + (cdisp - 5.0 * cdiss + 7.0 / 12.0) * ujp1
        + (-1.5 * cdisp + 2.5 * cdiss - 1.0 / 12.0) * ujp2
        + (0.5 * cdisp - 0.5 * cdiss) * ujp3
    )


def rhs_from_interface_flux(u: np.ndarray, dx: float, cdisp: np.ndarray, cdiss: np.ndarray) -> np.ndarray:
    flux_jph = six_point_interface_flux(u, cdisp, cdiss)
    # flux_jph[j] is j+1/2, flux_jph[j-1] is j-1/2.
    return -(flux_jph - periodic_shift(flux_jph, -1)) / dx


def rhs_mdcd(u: np.ndarray, dx: float) -> np.ndarray:
    cdisp = np.full_like(u, 0.0463783, dtype=float)
    cdiss = np.full_like(u, 0.001, dtype=float)
    return rhs_from_interface_flux(u, dx, cdisp, cdiss)


# -----------------------------------------------------------------------------
# SA-DRP sensor and local parameters
# -----------------------------------------------------------------------------

def sa_drp_local_parameters(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k_ESW, c_disp, c_diss) at interfaces j+1/2.

    This is a direct implementation of Li & Ren (2023), Eq. (34), Eq. (42),
    and Eq. (45), including the small-amplitude treatment below Eq. (45).
    """
    fjm2 = periodic_shift(u, -2)
    fjm1 = periodic_shift(u, -1)
    fj = u
    fjp1 = periodic_shift(u, 1)
    fjp2 = periodic_shift(u, 2)
    fjp3 = periodic_shift(u, 3)

    s1 = fjp1 - 2.0 * fj + fjm1
    s2 = (fjp2 - 2.0 * fj + fjm2) / 4.0
    s3 = fjp2 - 2.0 * fjp1 + fj
    s4 = (fjp3 - 2.0 * fjp1 + fjm1) / 4.0
    c1 = fjp1 - fj
    c2 = (fjp2 - fjm1) / 3.0

    numerator = (
        np.abs(np.abs(s1 + s2) - np.abs(s1 - s2))
        + np.abs(np.abs(s3 + s4) - np.abs(s3 - s4))
        + np.abs(np.abs(c1 + c2) - 0.5 * np.abs(c1 - c2))
        + 2.0 * EPS_SENSOR
    )
    denominator = (
        np.abs(s1 + s2)
        + np.abs(s1 - s2)
        + np.abs(s3 + s4)
        + np.abs(s3 - s4)
        + np.abs(c1 + c2)
        + np.abs(c1 - c2)
        + EPS_SENSOR
    )

    ratio = np.minimum(numerator / denominator, 1.0)
    arg = np.clip(2.0 * ratio - 1.0, -1.0, 1.0)
    k_esw = np.arccos(arg)

    cdisp = np.empty_like(u, dtype=float)
    low = (0.0 <= k_esw) & (k_esw < 0.01)
    mid = (0.01 <= k_esw) & (k_esw < 2.5)
    high = ~(low | mid)
    cdisp[low] = 1.0 / 30.0
    kk = k_esw[mid]
    cdisp[mid] = (
        kk + (1.0 / 6.0) * np.sin(2.0 * kk) - (4.0 / 3.0) * np.sin(kk)
    ) / (
        np.sin(3.0 * kk) - 4.0 * np.sin(2.0 * kk) + 5.0 * np.sin(kk)
    )
    cdisp[high] = 0.1985842

    cdiss = np.empty_like(u, dtype=float)
    diss_low = k_esw <= 1.0
    diss_high = ~diss_low
    cdiss[diss_low] = 0.001
    cdiss[diss_high] = np.minimum(
        0.001 + 0.011 * np.sqrt(np.maximum((k_esw[diss_high] - 1.0) / (math.pi - 1.0), 0.0)),
        0.012,
    )

    local_max = np.maximum.reduce([fjm2, fjm1, fj, fjp1, fjp2, fjp3])
    local_min = np.minimum.reduce([fjm2, fjm1, fj, fjp1, fjp2, fjp3])
    small_amplitude = (local_max - local_min) < SMALL_AMPLITUDE_THRESHOLD
    k_esw = np.where(small_amplitude, 0.0, k_esw)
    cdisp = np.where(small_amplitude, 1.0 / 30.0, cdisp)
    cdiss = np.where(small_amplitude, 0.001, cdiss)

    return k_esw, cdisp, cdiss


def rhs_sa_drp(u: np.ndarray, dx: float, frozen_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if frozen_parameters is None:
        frozen_parameters = sa_drp_local_parameters(u)
    _, cdisp, cdiss = frozen_parameters
    return rhs_from_interface_flux(u, dx, cdisp, cdiss), frozen_parameters


# -----------------------------------------------------------------------------
# RK4 time marching
# -----------------------------------------------------------------------------

def rk4_step(u: np.ndarray, dt: float, dx: float, scheme: str) -> np.ndarray:
    if scheme == "DRP":
        rhs = lambda v: rhs_drp(v, dx)
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
    elif scheme == "DRP-M":
        rhs = lambda v: rhs_drp_m(v, dx)
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
    elif scheme == "MDCD":
        rhs = lambda v: rhs_mdcd(v, dx)
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
    elif scheme == "SA-DRP":
        # Li & Ren state that the sensor and parameters are computed only at
        # the first RK stage and then reused in the following three stages.
        params = sa_drp_local_parameters(u)
        k1, _ = rhs_sa_drp(u, dx, params)
        k2, _ = rhs_sa_drp(u + 0.5 * dt * k1, dx, params)
        k3, _ = rhs_sa_drp(u + 0.5 * dt * k2, dx, params)
        k4, _ = rhs_sa_drp(u + dt * k3, dx, params)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve(m: int, n_grid: int, scheme: str) -> RunResult:
    x = np.arange(n_grid, dtype=float) / float(n_grid)
    dx = 1.0 / float(n_grid)
    nominal_dt = CFL * dx / abs(A)
    # Adjust the last step so that the comparison is exactly at t=10.
    nsteps = int(math.ceil(T_END / nominal_dt))
    dt = T_END / float(nsteps)

    u = wave_packet(x, m)
    for _ in range(nsteps):
        u = rk4_step(u, dt, dx, scheme)
    exact = exact_solution(x, T_END, m)
    return RunResult(x=x, u=u, exact=exact, l2=l2_error(u, exact), dt=dt, nsteps=nsteps)


def validate_sensor() -> pd.DataFrame:
    rows = []
    n = 256
    x = np.arange(n, dtype=float) / n
    for ell in [1, 5, 10, 20, 40, 80]:
        u = np.sin(2.0 * math.pi * ell * x)
        k_esw, _, _ = sa_drp_local_parameters(u)
        exact_k = 2.0 * math.pi * ell / n
        rows.append(
            {
                "mode": ell,
                "exact_scaled_wavenumber": exact_k,
                "mean_k_ESW": float(k_esw.mean()),
                "max_abs_error": float(np.max(np.abs(k_esw - exact_k))),
            }
        )
    return pd.DataFrame(rows)


def generate_all(output_root: Path) -> None:
    fig_dir = output_root / "figures"
    data_dir = output_root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) convergence data for Fig.13-like plot
    conv_rows = []
    for m in CONVERGENCE_M:
        for n in PAPER_CONVERGENCE_N:
            for scheme in SCHEME_ORDER:
                r = solve(m, n, scheme)
                conv_rows.append({"m": m, "N": n, "scheme": scheme, "L2_error": r.l2, "dt": r.dt, "nsteps": r.nsteps})
    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(data_dir / "q3_convergence_results.csv", index=False)

    # 2) fixed N=256 extended m data
    ext_rows = []
    for m in EXTENDED_M:
        for scheme in SCHEME_ORDER:
            r = solve(m, 256, scheme)
            ext_rows.append({"m": m, "N": 256, "scheme": scheme, "L2_error": r.l2, "dt": r.dt, "nsteps": r.nsteps})
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(data_dir / "q3_error_results.csv", index=False)

    # 3) sensor validation
    validate_sensor().to_csv(data_dir / "q3_sensor_validation.csv", index=False)

    plot_convergence(conv_df, fig_dir / "q3_convergence_m5_m10_m15_m20.png")
    plot_extended_m(ext_df, fig_dir / "q3_extended_m_study.png")
    plot_snapshot(fig_dir / "q3_snapshot_N256_m20.png")


def _plot_reference_slopes(ax, m: int) -> None:
    # Continuous abscissa for visual guide lines. These are reference slopes only;
    # no numerical data are modified.
    xref_full = np.geomspace(50.0, 1000.0, 300)
    if m == 5:
        y4 = 0.60 * (xref_full / 50.0) ** -4
        y5 = 2.5e-4 * (xref_full / 50.0) ** -5
        ax.set_ylim(1e-10, 1e0)
        ax.loglog(xref_full, y4, 'k--', linewidth=1.6, label='4th')
        ax.loglog(xref_full, y5, color='black', linestyle=(0, (4, 4)), linewidth=1.6, label='5th')
    elif m == 10:
        # Lift the upper 4th-order guide line slightly.
        y4 = 18.0 * (xref_full / 50.0) ** -4
        y5 = 1.6e-2 * (xref_full / 50.0) ** -5
        ax.set_ylim(1e-10, 1e2)
        ax.loglog(xref_full, y4, 'k--', linewidth=1.6, label='4th')
        ax.loglog(xref_full, y5, color='black', linestyle=(0, (4, 4)), linewidth=1.6, label='5th')
    elif m == 15:
        # Lift the upper 4th-order guide line a little higher.
        y4 = 25.0 * (xref_full / 50.0) ** -4
        y5 = 2.0e-2 * (xref_full / 50.0) ** -5
        ax.set_ylim(1e-8, 1e2)
        ax.loglog(xref_full, y4, 'k--', linewidth=1.6, label='4th')
        ax.loglog(xref_full, y5, color='black', linestyle=(0, (4, 4)), linewidth=1.6, label='5th')
    else:
        # For m=20, keep the x-start at the left axis edge and place the upper
        # 4th-order guide line so that its y-start is about 10^2.
        y4 = 100.0 * (xref_full / 50.0) ** -4
        y5 = 1.0e-1 * (xref_full / 50.0) ** -5
        ax.set_ylim(1e-8, 1e2)
        ax.loglog(xref_full, y4, 'k--', linewidth=1.6, label='4th')
        ax.loglog(xref_full, y5, color='black', linestyle=(0, (4, 4)), linewidth=1.6, label='5th')


def plot_convergence(df: pd.DataFrame, out_path: Path) -> None:
    colors = {"DRP": "red", "DRP-M": "limegreen", "MDCD": "blue", "SA-DRP": "magenta"}
    markers = {"DRP": "s", "DRP-M": "v", "MDCD": "^", "SA-DRP": "o"}

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    axes = axes.ravel()
    for ax, m, tag in zip(axes, CONVERGENCE_M, ["(a)", "(b)", "(c)", "(d)"]):
        sub = df[df["m"] == m]
        for scheme in SCHEME_ORDER:
            s = sub[sub["scheme"] == scheme].sort_values("N")
            ax.loglog(
                s["N"],
                s["L2_error"],
                marker=markers[scheme],
                color=colors[scheme],
                linewidth=2.0,
                markersize=7.5,
                markerfacecolor="white",
                markeredgewidth=1.8,
                label=scheme,
            )
        _plot_reference_slopes(ax, m)
        ax.set_xlim(50, 1000)
        ax.set_xlabel("N")
        ax.set_ylabel("L2 error")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True, fancybox=False, edgecolor="black")
        ax.text(0.5, -0.24, f"{tag}  m={m}", transform=ax.transAxes, ha="center", va="top", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot(out_path: Path) -> None:
    n = 256
    m = 20
    exact_x = np.arange(n, dtype=float) / n
    exact = exact_solution(exact_x, T_END, m)

    results: Dict[str, RunResult] = {scheme: solve(m, n, scheme) for scheme in SCHEME_ORDER}
    colors = {"DRP": "red", "DRP-M": "darkorange", "MDCD": "limegreen", "SA-DRP": "blue"}
    markers = {"DRP": "s", "DRP-M": "v", "MDCD": "^", "SA-DRP": "o"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, xlim in zip(axes, [(0.0, 1.0), (0.8, 1.0)]):
        ax.plot(exact_x, exact, color="black", linewidth=1.8, label="Exact")
        for scheme in SCHEME_ORDER:
            r = results[scheme]
            ax.plot(r.x, r.u, linestyle="None", marker=markers[scheme], markersize=3.8, markerfacecolor="white", markeredgewidth=1.0, color=colors[scheme], label=scheme)
        ax.set_xlim(*xlim)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_title("(a) distributions of solution at t=10.0")
    axes[1].set_title("(b) enlarged portion near the right boundary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_extended_m(df: pd.DataFrame, out_path: Path) -> None:
    colors = {"DRP": "red", "DRP-M": "limegreen", "MDCD": "blue", "SA-DRP": "magenta"}
    markers = {"DRP": "s", "DRP-M": "v", "MDCD": "^", "SA-DRP": "o"}
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for scheme in SCHEME_ORDER:
        s = df[df["scheme"] == scheme].sort_values("m")
        ax.plot(s["m"], s["L2_error"], marker=markers[scheme], color=colors[scheme], linewidth=2.0, markersize=7.5, markerfacecolor="white", markeredgewidth=1.8, label=scheme)
    ax.set_xlabel("m")
    ax.set_ylabel("L2 error")
    ax.set_xlim(18, 122)
    ax.set_ylim(0, 0.13)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_all(Path(__file__).resolve().parents[1])
