import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20260420
EPSILON = 0.1
K0 = 24
KMAX = 64
DOMAIN = (0.0, 1.0)
A = 1.0
CFL = 0.2
T_END = 1.0  # one period on [0,1] for a=1

OUTDIR = Path('/mnt/data/problem4_out')
OUTDIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Deterministic random phases
# --------------------------------------------------
rng = np.random.default_rng(SEED)
PSI = rng.random(KMAX + 1)  # use indices 1..KMAX


def energy_spectrum(k: int, k0: float = K0) -> float:
    return (k / k0) ** 4 * math.exp(-2.0 * (k / k0) ** 2)


def initial_condition(x: np.ndarray) -> np.ndarray:
    u = np.ones_like(x)
    for k in range(1, KMAX + 1):
        u += EPSILON * math.sqrt(energy_spectrum(k)) * np.sin(2.0 * math.pi * k * (x + PSI[k]))
    return u


def exact_solution(x: np.ndarray, t: float) -> np.ndarray:
    x_shift = (x - A * t) % 1.0
    return initial_condition(x_shift)


def periodic_roll(u, shift):
    return np.roll(u, shift)


def l1_error(u, ue):
    return float(np.mean(np.abs(u - ue)))


# DRP derivatives

def derivative_antisymmetric(u: np.ndarray, dx: float, a1: float, a2: float, a3: float) -> np.ndarray:
    return (
        a1 * (periodic_roll(u, -1) - periodic_roll(u, 1))
        + a2 * (periodic_roll(u, -2) - periodic_roll(u, 2))
        + a3 * (periodic_roll(u, -3) - periodic_roll(u, 3))
    ) / dx


def rhs_drp(u: np.ndarray, dx: float) -> np.ndarray:
    return -A * derivative_antisymmetric(u, dx, 0.79926643, -0.18941314, 0.02651995)


def rhs_drp_m(u: np.ndarray, dx: float) -> np.ndarray:
    return -A * derivative_antisymmetric(u, dx, 0.770882380518, -0.166705904415, 0.020843142770)


# Upwind finite differences for positive a

def rhs_upwind1(u: np.ndarray, dx: float) -> np.ndarray:
    dudx = (u - periodic_roll(u, 1)) / dx
    return -A * dudx


def rhs_upwind2(u: np.ndarray, dx: float) -> np.ndarray:
    dudx = (3*u - 4*periodic_roll(u,1) + periodic_roll(u,2)) / (2*dx)
    return -A * dudx


def rhs_upwind3(u: np.ndarray, dx: float) -> np.ndarray:
    dudx = (11*u - 18*periodic_roll(u,1) + 9*periodic_roll(u,2) - 2*periodic_roll(u,3)) / (6*dx)
    return -A * dudx


# MDCD / SA-DRP

def mdcd_flux(u: np.ndarray, cdisp: np.ndarray, cdiss: np.ndarray) -> np.ndarray:
    um2 = periodic_roll(u, 2)
    um1 = periodic_roll(u, 1)
    u0 = u
    up1 = periodic_roll(u, -1)
    up2 = periodic_roll(u, -2)
    up3 = periodic_roll(u, -3)
    return (
        (0.5 * cdisp + 0.5 * cdiss) * um2
        + (-1.5 * cdisp - 2.5 * cdiss - 1.0 / 12.0) * um1
        + (cdisp + 5.0 * cdiss + 7.0 / 12.0) * u0
        + (cdisp - 5.0 * cdiss + 7.0 / 12.0) * up1
        + (-1.5 * cdisp + 2.5 * cdiss - 1.0 / 12.0) * up2
        + (0.5 * cdisp - 0.5 * cdiss) * up3
    )


def mdcd_rhs_with_params(u: np.ndarray, dx: float, cdisp: np.ndarray, cdiss: np.ndarray) -> np.ndarray:
    flux = mdcd_flux(u, cdisp, cdiss)
    return -A * (flux - periodic_roll(flux, 1)) / dx


def rhs_mdcd(u: np.ndarray, dx: float, cdisp_const=0.0463783, cdiss_const=0.001) -> np.ndarray:
    cdisp = np.full_like(u, cdisp_const)
    cdiss = np.full_like(u, cdiss_const)
    return mdcd_rhs_with_params(u, dx, cdisp, cdiss)


def sa_drp_parameters(u: np.ndarray):
    eps = 1.0e-8
    ujm2 = periodic_roll(u, 2)
    ujm1 = periodic_roll(u, 1)
    uj = u
    ujp1 = periodic_roll(u, -1)
    ujp2 = periodic_roll(u, -2)
    ujp3 = periodic_roll(u, -3)

    S1 = ujp1 - 2.0 * uj + ujm1
    S2 = (ujp2 - 2.0 * uj + ujm2) / 4.0
    S3 = ujp2 - 2.0 * ujp1 + uj
    S4 = (ujp3 - 2.0 * ujp1 + ujm1) / 4.0
    C1 = ujp1 - uj
    C2 = (ujp2 - ujm1) / 3.0

    numerator = (
        np.abs(S1 + S2) - np.abs(S1 - S2)
        + np.abs(S3 + S4) - np.abs(S3 - S4)
        + np.abs(C1 + C2) - 0.5 * np.abs(C1 - C2)
        + 2.0 * eps
    )
    denominator = (
        np.abs(S1 + S2) + np.abs(S1 - S2)
        + np.abs(S3 + S4) + np.abs(S3 - S4)
        + np.abs(C1 + C2) + np.abs(C1 - C2)
        + eps
    )
    arg = np.clip(np.minimum(2.0 * numerator / denominator, 1.0), -1.0, 1.0)
    k_esw = np.arccos(arg)

    stencil_max = np.maximum.reduce([ujm2, ujm1, uj, ujp1, ujp2, ujp3])
    stencil_min = np.minimum.reduce([ujm2, ujm1, uj, ujp1, ujp2, ujp3])
    small_amp = (stencil_max - stencil_min) < 1.0e-3

    cdisp = np.empty_like(u)
    mask1 = (k_esw >= 0.0) & (k_esw < 0.01)
    mask2 = (k_esw >= 0.01) & (k_esw < 2.5)
    mask3 = ~(mask1 | mask2)
    cdisp[mask1] = 1.0 / 30.0
    k2 = k_esw[mask2]
    cdisp[mask2] = (k2 + (1.0/6.0)*np.sin(2*k2) - (4.0/3.0)*np.sin(k2)) / (np.sin(3*k2) - 4*np.sin(2*k2) + 5*np.sin(k2))
    cdisp[mask3] = 0.1985842

    cdiss = np.empty_like(u)
    low = k_esw <= 1.0
    high = ~low
    cdiss[low] = 0.001
    cdiss[high] = np.minimum(0.001 + 0.011 * np.sqrt(np.maximum((k_esw[high]-1.0)/(math.pi-1.0), 0.0)), 0.012)

    k_esw = np.where(small_amp, 0.0, k_esw)
    cdisp = np.where(small_amp, 1.0/30.0, cdisp)
    cdiss = np.where(small_amp, 0.001, cdiss)
    return k_esw, cdisp, cdiss


def rhs_sa_drp(u: np.ndarray, dx: float, cached=None):
    params = sa_drp_parameters(u) if cached is None else cached
    _, cdisp, cdiss = params
    return mdcd_rhs_with_params(u, dx, cdisp, cdiss), params


SCHEMES = {
    'DRP': lambda u, dx: rhs_drp(u, dx),
    'DRP-M': lambda u, dx: rhs_drp_m(u, dx),
    'MDCD': lambda u, dx: rhs_mdcd(u, dx),
    'UW1': lambda u, dx: rhs_upwind1(u, dx),
    'UW2': lambda u, dx: rhs_upwind2(u, dx),
    'UW3': lambda u, dx: rhs_upwind3(u, dx),
}


def advance_rk4(u0: np.ndarray, dx: float, dt: float, nsteps: int, scheme: str) -> np.ndarray:
    u = u0.copy()
    for _ in range(nsteps):
        if scheme == 'SA-DRP':
            params = sa_drp_parameters(u)
            k1, _ = rhs_sa_drp(u, dx, params)
            k2, _ = rhs_sa_drp(u + 0.5*dt*k1, dx, params)
            k3, _ = rhs_sa_drp(u + 0.5*dt*k2, dx, params)
            k4, _ = rhs_sa_drp(u + dt*k3, dx, params)
        else:
            rhs = SCHEMES[scheme]
            k1 = rhs(u, dx)
            k2 = rhs(u + 0.5*dt*k1, dx)
            k3 = rhs(u + 0.5*dt*k2, dx)
            k4 = rhs(u + dt*k3, dx)
        u = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return u


def solve(n: int, scheme: str):
    x0, x1 = DOMAIN
    x = np.linspace(x0, x1, n, endpoint=False)
    dx = (x1 - x0) / n
    dt_nom = CFL * dx / abs(A)
    nsteps = math.ceil(T_END / dt_nom)
    dt = T_END / nsteps
    u0 = initial_condition(x)
    u = advance_rk4(u0, dx, dt, nsteps, scheme)
    ue = exact_solution(x, T_END)
    return x, u, ue, l1_error(u, ue)


def convergence_orders(errors: List[float]):
    orders = [np.nan]
    for i in range(1, len(errors)):
        orders.append(math.log(errors[i-1]/errors[i], 2))
    return orders


def make_snapshot_plot(n=256):
    x, ue, _, _ = solve(n, 'DRP')
    # exact from separate call to avoid confusion
    x = np.linspace(0.0,1.0,n,endpoint=False)
    ue = exact_solution(x, T_END)
    plt.figure(figsize=(10,5.2))
    plt.plot(x, ue, 'k-', linewidth=2.2, label='Exact')
    for s in ['DRP','DRP-M','MDCD','SA-DRP']:
        _, u, _, _ = solve(n, s)
        plt.plot(x, u, linewidth=1.2, label=s)
    plt.xlim(0,1)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Q4 snapshot at t={T_END}, N={n}, CFL={CFL}')
    plt.legend(ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'q4_snapshot_N256.png', dpi=220)
    plt.close()

    # zoom view
    plt.figure(figsize=(10,5.2))
    plt.plot(x, ue, 'k-', linewidth=2.2, label='Exact')
    for s in ['DRP','DRP-M','MDCD','SA-DRP']:
        _, u, _, _ = solve(n, s)
        plt.plot(x, u, linewidth=1.2, label=s)
    plt.xlim(0.68,0.88)
    ymin, ymax = ue[(x>=0.68)&(x<=0.88)].min(), ue[(x>=0.68)&(x<=0.88)].max()
    pad = 0.08*(ymax-ymin)
    plt.ylim(ymin-pad, ymax+pad)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Enlarged view')
    plt.legend(ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'q4_snapshot_zoom_N256.png', dpi=220)
    plt.close()


def run_convergence():
    Ns = [64,128,256,512,1024]
    rows = []
    for s in ['DRP','DRP-M','MDCD','SA-DRP']:
        errs = []
        for n in Ns:
            _, _, _, err = solve(n, s)
            errs.append(err)
        orders = convergence_orders(errs)
        for n,e,p in zip(Ns, errs, orders):
            rows.append({'scheme':s,'N':n,'L1_error':e,'observed_order':p})
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / 'q4_convergence.csv', index=False)

    plt.figure(figsize=(8.8,5.8))
    for s in ['DRP','DRP-M','MDCD','SA-DRP']:
        sub = df[df.scheme==s]
        plt.loglog(sub['N'], sub['L1_error'], marker='o', linewidth=1.8, label=s)
    plt.xlabel('N')
    plt.ylabel(r'$L_1$ error')
    plt.title('Q4 convergence curves')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'q4_convergence.png', dpi=220)
    plt.close()
    return df


def find_min_n(scheme: str, threshold=1e-3, n_start=64, n_step=64, n_max=4096):
    best = None
    history = []
    for n in range(n_start, n_max + 1, n_step):
        _, _, _, err = solve(n, scheme)
        history.append((n, err))
        if err <= threshold:
            best = n
            break
    return best, history


def run_resolution():
    rows = []
    histories = {}
    for s in ['UW1','UW2','UW3','DRP','DRP-M','MDCD','SA-DRP']:
        best, hist = find_min_n(s)
        histories[s] = hist
        rows.append({'scheme':s, 'min_N_for_L1<=1e-3': best if best is not None else '>4096'})
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / 'q4_resolution_threshold.csv', index=False)

    plt.figure(figsize=(8.8,5.8))
    for s, hist in histories.items():
        x = [a for a,b in hist]
        y = [b for a,b in hist]
        plt.semilogy(x, y, marker='o', markersize=3.5, linewidth=1.2, label=s)
    plt.axhline(1e-3, color='k', linestyle='--', linewidth=1.2)
    plt.xlabel('N')
    plt.ylabel(r'$L_1$ error')
    plt.title('Resolution study and threshold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'q4_resolution.png', dpi=220)
    plt.close()
    return df, histories


def write_summary(df_conv: pd.DataFrame, df_res: pd.DataFrame):
    lines = []
    lines.append('Q4 numerical experiment summary\n')
    lines.append(f'SEED = {SEED}\n')
    lines.append(f'Initial condition: u(x,0)=1+epsilon*sum_{{k=1}}^{{64}} sqrt(E(k))*sin(2*pi*k*(x+psi_k)), epsilon={EPSILON}, k0={K0}\n')
    lines.append(f'E(k)=(k/k0)^4 * exp(-2*(k/k0)^2), x in [0,1], periodic BC, CFL={CFL}, t_end={T_END}\n\n')
    lines.append('Convergence table:\n')
    lines.append(df_conv.to_string(index=False))
    lines.append('\n\nResolution threshold table:\n')
    lines.append(df_res.to_string(index=False))
    (OUTDIR / 'q4_summary.txt').write_text(''.join(lines), encoding='utf-8')


if __name__ == '__main__':
    make_snapshot_plot()
    df_conv = run_convergence()
    df_res, histories = run_resolution()
    write_summary(df_conv, df_res)
    print(df_conv)
    print(df_res)
