import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/mnt/data/q2_outputs')
OUT.mkdir(exist_ok=True)

alpha, beta, theta = sp.symbols('alpha beta theta', real=True)

coeff = {
    -3: -sp.Rational(1,2)*alpha - sp.Rational(1,2)*beta,
    -2: 2*alpha + 3*beta + sp.Rational(1,12),
    -1: -sp.Rational(5,2)*alpha - sp.Rational(15,2)*beta - sp.Rational(2,3),
    0: 10*beta,
    1: sp.Rational(5,2)*alpha - sp.Rational(15,2)*beta + sp.Rational(2,3),
    2: -2*alpha + 3*beta - sp.Rational(1,12),
    3: sp.Rational(1,2)*alpha - sp.Rational(1,2)*beta,
}

def moments(maxq=8):
    return [sp.expand(sum(v*(k**q) for k,v in coeff.items())) for q in range(maxq+1)]

def kstar_expr():
    sym = sum(v*sp.exp(sp.I*k*theta) for k,v in coeff.items())
    return sp.simplify(-sp.I*sym)

KM = sp.expand_complex(kstar_expr())
REAL = sp.simplify(sp.re(KM))
IMAG = sp.simplify(sp.im(KM))

# Save symbolic results
with open(OUT/'symbolic_results.txt','w',encoding='utf-8') as f:
    ms = moments(8)
    for q,m in enumerate(ms):
        f.write(f'M_{q} = {sp.simplify(m)}\n')
    f.write('\nRe(k*) = %s\n' % REAL)
    f.write('Im(k*) = %s\n' % IMAG)

# Numerical functions
re_func = sp.lambdify((theta, alpha), REAL.subs(beta, 0), 'numpy')
imag_func = sp.lambdify((theta, beta), IMAG.subs(alpha, 0), 'numpy')

th = np.linspace(0, np.pi, 1200)

# Optimal alpha values using weighted objective
import mpmath as mp

def alpha_star(t):
    g = lambda x: 4/3*mp.sin(x) - 1/6*mp.sin(2*x) - x
    h = lambda x: 5*mp.sin(x) - 4*mp.sin(2*x) + mp.sin(3*x)
    w = lambda x: mp.e**(t*(mp.pi - x))
    num = mp.quad(lambda x: w(x)*g(x)*h(x), [0, mp.pi])
    den = mp.quad(lambda x: w(x)*h(x)*h(x), [0, mp.pi])
    return float(-num/den)

opt_table = [(t, alpha_star(t)) for t in [4,6,8,10]]
with open(OUT/'alpha_opt_table.txt','w',encoding='utf-8') as f:
    for t,a in opt_table:
        f.write(f't = {t:2d}, alpha* = {a:.10f}\n')

alpha_ref = dict(opt_table)[8]

# figure 1: Re(k*) under various alpha, beta=0
plt.figure(figsize=(6.6,4.6))
plt.plot(th, th, lw=2.0, label='exact: $k\\Delta x$')
for a,lbl in [(0.0, r'$\alpha=0$'), (1/30, r'$\alpha=1/30$'), (alpha_ref, r'$\alpha=0.0463783$'), (0.0714071, r'$\alpha=0.0714071$')]:
    plt.plot(th, re_func(th, a), lw=1.8, label=lbl)
plt.xlabel(r'$\kappa=k\Delta x$')
plt.ylabel(r'$\Re(k^*\Delta x)$')
plt.title('Dispersion curves for different alpha (beta = 0)')
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT/'dispersion_alpha.pdf')
plt.savefig(OUT/'dispersion_alpha.png', dpi=220)
plt.close()

# figure 2: Im(k*) under various beta, alpha fixed
plt.figure(figsize=(6.6,4.6))
for b,lbl in [(0.0, r'$\beta=0$'), (0.002, r'$\beta=0.002$'), (0.005, r'$\beta=0.005$'), (0.01, r'$\beta=0.01$')]:
    plt.plot(th, imag_func(th, b), lw=1.8, label=lbl)
plt.axhline(0,color='k',lw=1)
plt.xlabel(r'$\kappa=k\Delta x$')
plt.ylabel(r'$\Im(k^*\Delta x)$')
plt.title('Dissipation curves for different beta')
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT/'dissipation_beta.pdf')
plt.savefig(OUT/'dissipation_beta.png', dpi=220)
plt.close()

# figure 3: verify Re(k*) independent of beta
plt.figure(figsize=(6.6,4.6))
for b,lbl in [(0.0, r'$\beta=0$'), (0.002, r'$\beta=0.002$'), (0.005, r'$\beta=0.005$')]:
    expr = sp.lambdify(theta, REAL.subs({alpha: alpha_ref, beta: b}), 'numpy')
    plt.plot(th, expr(th), lw=1.8, label=lbl)
plt.plot(th, th, 'k--', lw=1.2, label='exact')
plt.xlabel(r'$\kappa=k\Delta x$')
plt.ylabel(r'$\Re(k^*\Delta x)$')
plt.title('Identical dispersion curves for different beta')
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT/'dispersion_beta_independence.png', dpi=220)
plt.close()

# figure 4: verify Im(k*) independent of alpha when beta fixed
plt.figure(figsize=(6.6,4.6))
for a,lbl in [(0.0, r'$\alpha=0$'), (1/30, r'$\alpha=1/30$'), (alpha_ref, r'$\alpha=0.0463783$')]:
    expr = sp.lambdify(theta, IMAG.subs({alpha: a, beta: 0.005}), 'numpy')
    plt.plot(th, expr(th), lw=1.8, label=lbl)
plt.axhline(0,color='k',lw=1)
plt.xlabel(r'$\kappa=k\Delta x$')
plt.ylabel(r'$\Im(k^*\Delta x)$')
plt.title('Identical dissipation curves for different alpha')
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT/'dissipation_alpha_independence.png', dpi=220)
plt.close()

print('done')
